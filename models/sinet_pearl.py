import torch
import torch.nn as nn
import copy

from models.vit_pearl import VisionTransformer, PatchEmbed, Block, resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.zoo import CodaPrompt

class ViT_lora_co(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank)


    def forward(self, x, task_id):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id)

        x = self.norm(x)
        
        return x, prompt_loss



def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_lora_co, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model



class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, n_tasks=args["num_tasks"])
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224_in21k', pretrained=True, **model_kwargs)
        # print(self.image_encoder)
        # exit()

        self.class_num = 0

        # Initialize the classifier pool
        self.classifier_pool = nn.ModuleList()

        # self.prompt_pool = CodaPrompt(args["embd_dim"], args["total_sessions"], args["prompt_param"])

        self.numtask = -1

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image, task=None):
        if task == None:
            image_features, _ = self.image_encoder(image, self.numtask-1)
        else:
            image_features, _ = self.image_encoder(image, task)
        image_features = image_features[:,0,:]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, task_id):

        logits = []
        image_features, prompt_loss = self.image_encoder(image, task_id)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        classifier = self.classifier_pool[task_id]
        logits = classifier(image_features)

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            'prompt_loss': prompt_loss
        }
    
    def update_task(self, _cur_class_num):
        self.numtask += 1
        self.class_num += _cur_class_num

    def add_classifier(self, num_cls):
        self.classifier_pool.append(nn.Linear(self.feature_dim, num_cls, bias=True))

    def interface(self, image):
        """
        Class-IL inference: Gets logits from all learned tasks and concatenates them.
        Each task uses the shared backbone + its specific LoRA parameters.
        """
        self.image_encoder.eval() # Ensure encoder is in eval mode
        all_logits = []
        
        # self.numtask represents the number of tasks learned (e.g., if tasks 0, 1 are learned, self.numtask = 2)
        # Iterate from task_idx 0 to self.numtask - 1
        for task_idx in range(self.numtask):
            # Get features specific to this task_idx by activating its LoRAs in the image_encoder
            # The image_encoder's forward method should handle applying the correct LoRA based on task_id
            task_specific_features, _ = self.image_encoder(image, task_id=task_idx)
            
            # Assuming CLS token is at index 0
            task_specific_features = task_specific_features[:, 0, :] 
            task_specific_features = task_specific_features.view(task_specific_features.size(0), -1)

            # Apply the classifier for this specific task
            # Ensure self.classifier_pool[task_idx] is the trained classifier for task_idx
            # and its output dimension matches the number of classes in task_idx.
            if task_idx < len(self.classifier_pool) and self.classifier_pool[task_idx] is not None:
                logits_for_task = self.classifier_pool[task_idx](task_specific_features)
                all_logits.append(logits_for_task)
            else:
                # This case should ideally not be reached if classifiers are managed correctly
                # Or, if a task has no classes (e.g. a dummy task 0 before real tasks start)
                # For robust handling, you might want to log a warning or ensure self.numtask aligns with available classifiers
                pass

        if not all_logits:
            # Handle case where no logits were generated (e.g., before any task is learned)
            # Returning a zero tensor or raising an error might be appropriate depending on desired behavior.
            # For now, let's assume at least one task is learned.
            # This part depends on how you handle the very first state (no tasks learned).
            # If num_tasks can be 0, you need to decide what to return.
            # Example: return torch.empty(image.size(0), 0).to(image.device)
            # For now, let's assume self.numtask >= 1 for this to be called in eval
            if image.size(0) > 0 :
                 # Fallback, though ideally, this should not happen if `self.numtask` is managed well.
                 # The number of classes for fallback is uncertain. Let's assume 0 output classes.
                 num_output_classes = 0 # Or some default if expected for an edge case
                 if self.numtask > 0 and len(self.classifier_pool) > 0 and self.classifier_pool[0] is not None:
                    num_output_classes = self.classifier_pool[0].out_features
                 else: # A very rough fallback if no classifier info is available
                    num_output_classes = self.args.get("init_cls", 1) if hasattr(self, 'args') else 1


                 return torch.zeros(image.size(0), num_output_classes * self.numtask if self.numtask > 0 else num_output_classes).to(image.device)
            else:
                 return torch.empty(0,0).to(image.device)


        # Concatenate logits from all task-specific classifiers along dimension 1
        concatenated_logits = torch.cat(all_logits, dim=1)
        return concatenated_logits

    def classifier_recall(self):
        self.classifier_pool.load_state_dict(self.old_state_dict)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
