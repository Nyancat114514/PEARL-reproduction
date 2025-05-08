import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_pearl import SiNet
from models.vit_pearl import Attention_LoRA
from copy import deepcopy
from utils.schedulers import CosineSchedule
import ipdb
import math

class PEARL(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))
        
        self.args = args

        self.E1_lr = args.get("E1_lr", 5e-3)
        self.E1_epoch = args.get("E1_epoch", 25)
        self.E2_lr = args.get("E2_lr", 5e-4)
        self.E2_epoch = args.get("E2_epoch", 25)
        self.pearl_min_rank = args.get("pearl_min_rank", 1)
        self.pearl_max_rank_ratio = args.get("pearl_max_rank_ratio", 0.5) # Ratio of embed_dim
        self.default_pearl_threshold_mean = args.get("default_pearl_threshold_mean", 0.1)
        self.default_pearl_rank = args.get("default_pearl_rank", 4)


        self.optim = args["optim"]
        self.batch_size = args["batch_size"]
        self.num_workers = args["num_workers"]

        self.topk = 1  # origin is 5
        
        self.debug = False
        self._cur_task = -1
        self._total_classes = 0
        self._known_classes = 0


    def after_task(self):
        # self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

        # Freeze LoRA parameters of the task just trained
        if self._cur_task >= 0:
            logging.info(f"Freezing LoRA parameters for task {self._cur_task}")
            for block in self._network.image_encoder.blocks:
                if isinstance(block.attn, Attention_LoRA):
                    if self._cur_task < len(block.attn.lora_A_k) and block.attn.lora_A_k[self._cur_task] is not None:
                        for param in block.attn.lora_A_k[self._cur_task].parameters():
                            param.requires_grad_(False)
                        for param in block.attn.lora_B_k[self._cur_task].parameters():
                            param.requires_grad_(False)

    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_task(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

        # if len(self._multiple_gpus) > 1:
        #     self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

    def _train_model_generic(self, model, train_loader, optimizer, scheduler, num_epochs, description, task_idx_for_loss):
        model.train()
        for epoch in range(num_epochs):
            losses = 0.
            correct, total = 0, 0
            prog_bar = tqdm(train_loader, desc=f"{description} Epoch {epoch+1}/{num_epochs}")
            for _, inputs, targets in prog_bar:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # Targets for current task are 0 to task_size-1
                # In data_manager, they are mapped from known_classes to total_classes-1
                # Loss needs targets relative to current task's output head.
                relative_targets = targets - self._known_classes 

                optimizer.zero_grad()
                # Model's forward might need current task_idx if it influences LoRA selection or head selection.
                # For E1, temp_model has only one head. For E2, _network.forward needs task_idx.
                if "Stage 3" in description or "Reference task" in description : # self._network
                    outputs = model(inputs, task_id=task_idx_for_loss) # Pass task_id to SiNet_PEARL
                else: # temp_model_for_E1
                    outputs = model(inputs, task_id=0) # Assumes temp_model_for_E1's forward doesn't need task_id

                logits = outputs['logits']

                loss = F.cross_entropy(logits, relative_targets)
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(relative_targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2) if total > 0 else 0
                prog_bar.set_postfix({'loss': loss.item(), 'acc': train_acc})

            if scheduler:
                scheduler.step()
            
            avg_loss = losses / len(train_loader) if len(train_loader) > 0 else 0
            avg_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2) if total > 0 else 0
            logging.info(f"{description} Epoch {epoch+1}/{num_epochs} => Avg Loss: {avg_loss:.3f}, Avg Accy: {avg_acc:.2f}")



    def _create_temp_model_for_E1(self):
        """
        Creates a temporary model for Stage 1 fine-tuning.
        This model should be a copy of the reference backbone (f_theta_r) 
        plus a classifier for the current task's classes.
        It should NOT include any LoRA parameters from previous or current tasks.
        """
        # Create a new SiNet_PEARL instance, which initializes VisionTransformer without LoRAs active yet
        temp_model = SiNet(self.args)
        
        # Load weights from the original backbone of self._network
        # This ensures we start from W^r for each layer
        main_net_state_dict = self._network.state_dict()
        
        temp_model_state_dict = temp_model.state_dict()
        
        # Copy only non-LoRA and non-classifier_pool weights
        for k_main, v_main in main_net_state_dict.items():
            if not ("lora_" in k_main or "classifier_pool" in k_main):
                    if k_main in temp_model_state_dict:
                        if temp_model_state_dict[k_main].shape == v_main.shape:
                            temp_model_state_dict[k_main].copy_(v_main)
                        else:
                            logging.warning(f"Shape mismatch for {k_main} in _create_temp_model_for_E1. Skipping.")
                    # else: it's a key from main network not in the base SiNet_PEARL structure (e.g. full head if not pooled)

        temp_model.load_state_dict(temp_model_state_dict, strict=False)

        # The temp_model already has a new classifier for one task (self.class_num classes)
        # from its __init__ and initial update_fc.
        # Ensure it has the correct number of output classes for the current new task.
        # SiNet_PEARL's update_fc handles adding a *new* classifier for the *current* task.
        # For temp_model, it should effectively only have *one* classifier for the *current new* classes.
        # So, we might need a specific classifier for it.
        temp_model.classifier_pool = nn.ModuleList([nn.Linear(temp_model.feature_dim, self._total_classes - self._known_classes)])
        temp_model.numtask = 0 # So it uses the first (and only) classifier in its pool

        return temp_model

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        # --- Parameter setup: Freeze all by default ---
        for param in self._network.parameters():
            param.requires_grad_(False)

        # --- PEARL Stages for new tasks (self._cur_task >= 0 if pretrained, or self._cur_task >= 1 if task 0 was ref) ---
        
        # Stage 1: Fine-tune a temporary copy of the reference network for E1 epochs
        logging.info(f"Task {self._cur_task} Stage 1: Fine-tuning temporary reference model.")
        temp_model_E1 = self._create_temp_model_for_E1().to(self._device)
        
        # All parameters of temp_model_E1 are trainable for this stage
        optimizer_E1 = optim.Adam(temp_model_E1.parameters(), lr=self.E1_lr)
        scheduler_E1 = CosineSchedule(optimizer=optimizer_E1, K=self.E1_epoch)
        
        self._train_model_generic(temp_model_E1, train_loader, optimizer_E1, scheduler_E1, 
                                  self.E1_epoch, f"Task {self._cur_task} Stage 1", task_idx_for_loss=0) # temp_model uses its 0-th head

        # Stage 2: Compute Task Vectors, Determine Rank, Initialize LoRA in self._network
        logging.info(f"Task {self._cur_task} Stage 2: Computing task vectors and initializing LoRA.")
        with torch.no_grad():
            original_network_eval_mode = self._network.training
            self._network.eval() # Ensure main network is in eval for consistent W_l^r access
            
            # Iterate through Attention_LoRA modules in self._network
            # Assuming self._network.blocks gives access to ViT blocks
            for block_idx, main_block in enumerate(self._network.image_encoder.blocks):
                if isinstance(main_block.attn, Attention_LoRA):
                    attn_module_main = main_block.attn
                    # Corresponding attention module from the E1-fine-tuned temporary model
                    attn_module_temp_E1 = temp_model_E1.image_encoder.blocks[block_idx].attn 

                    dim = attn_module_main.dim
                    
                    # W_k_r: Key projection from reference model (original qkv from main_block)
                    # qkv.weight is (out_features, in_features). For Linear, out_features = 3*dim, in_features = dim
                    W_qkv_r_all = attn_module_main.qkv.weight.data # Shape: (3*dim, dim)
                    W_k_r = W_qkv_r_all[dim : 2*dim, :]       # Shape: (dim, dim)

                    # W_k_t: Key projection from temp_model_E1
                    W_qkv_t_all = attn_module_temp_E1.qkv.weight.data # Shape: (3*dim, dim)
                    W_k_t = W_qkv_t_all[dim : 2*dim, :]       # Shape: (dim, dim)

                    W_c_k = W_k_t - W_k_r # Task vector for key projection W_cl^t

                    U, S_vector, Vh = torch.linalg.svd(W_c_k) # S_vector contains singular values

                    # Calculate dynamic threshold T (Eq. 3 from paper)
                    norm_W_c_k_sq = torch.sum(W_c_k * W_c_k)
                    norm_W_k_t_sq = torch.sum(W_k_t * W_k_t)
                    norm_W_k_r_sq = torch.sum(W_k_r * W_k_r)
                    
                    current_T_scalar_mean = self.default_pearl_threshold_mean
                    if (norm_W_k_t_sq + norm_W_k_r_sq).item() > 1e-9: # Avoid division by zero
                        dynamic_T_for_layer = norm_W_c_k_sq / (norm_W_k_t_sq + norm_W_k_r_sq)
                        current_T_scalar_mean = dynamic_T_for_layer.item()
                    
                    # Calculate dynamic rank k_l (Eq. 4 from paper)
                    rank_k_for_layer = self.default_pearl_rank
                    sum_S_vector_sq_total = torch.sum(S_vector * S_vector)
                    if sum_S_vector_sq_total.item() > 1e-9:
                        cumulative_explained_variance = torch.cumsum((S_vector * S_vector) / sum_S_vector_sq_total, dim=0)
                        met_threshold_indices = (cumulative_explained_variance >= current_T_scalar_mean).nonzero(as_tuple=False)
                        if len(met_threshold_indices) == 0:
                            rank_k_for_layer = len(S_vector) 
                        else:
                            rank_k_for_layer = met_threshold_indices[0].item() + 1
                    
                    max_rank_for_layer = int(attn_module_main.dim * self.pearl_max_rank_ratio)
                    rank_k_for_layer = max(self.pearl_min_rank, min(rank_k_for_layer, max_rank_for_layer))
                    
                    logging.info(f"  Block {block_idx} Key LoRA: W_c_k norm {norm_W_c_k_sq.item():.4f}, T_val {current_T_scalar_mean:.4f}, Determined Rank k: {rank_k_for_layer}")

                    # Add LoRA layers for the current task with determined rank
                    attn_module_main.add_lora_for_task(self._cur_task, rank=rank_k_for_layer)
                    
                    # Re-initialize these new LoRA weights (A_k, B_k for self._cur_task) as per PEARL paper's finding
                    attn_module_main.reinitialize_lora_for_task(self._cur_task, target_is_value=False)
            
            if original_network_eval_mode is False: # Set back to train if it was training
                 self._network.train()


        # Stage 3: Fine-tune self._network (with new LoRA_k for current task) for E2 epochs
        logging.info(f"Task {self._cur_task} Stage 3: Fine-tuning main network with new LoRA.")
        
        trainable_params_E2 = []
        # Unfreeze current task's LoRA_k parameters
        for block in self._network.image_encoder.blocks:
            if isinstance(block.attn, Attention_LoRA):
                attn_module = block.attn
                if self._cur_task < len(attn_module.lora_A_k) and attn_module.lora_A_k[self._cur_task] is not None:
                    for param in attn_module.lora_A_k[self._cur_task].parameters():
                        param.requires_grad_(True)
                        trainable_params_E2.append(param)
                    for param in attn_module.lora_B_k[self._cur_task].parameters():
                        param.requires_grad_(True)
                        trainable_params_E2.append(param)
        
        # Unfreeze current task's classifier head
        # SiNet_PEARL's update_fc adds a new classifier for self._cur_task at index self._cur_task
        self._network.add_classifier(self._total_classes - self._known_classes)
        current_classifier = self._network.classifier_pool[self._cur_task]
        current_classifier.to(self._device) 

        try:
            for param in current_classifier.parameters():
                param.requires_grad_(True)
                trainable_params_E2.append(param)
        except:
            logging.warning(f"PEARL Stage 3 (Task {self._cur_task}): No lora parameters found.")

        if not trainable_params_E2:
            logging.warning(f"PEARL Stage 3 (Task {self._cur_task}): No parameters found to train for E2 stage. Skipping.")
            return

        optimizer_E2 = optim.Adam(trainable_params_E2, lr=self.E2_lr)
        scheduler_E2 = CosineSchedule(optimizer=optimizer_E2, K=self.E2_epoch)

        self._train_model_generic(self._network, train_loader, optimizer_E2, scheduler_E2, 
                                  self.E2_epoch, f"Task {self._cur_task} Stage 3", task_idx_for_loss=self._cur_task)


    def _evaluate(self, y_pred, y_true):
        ret = {}
        
        print("len(y_pred):", len(y_pred), "len(y_true):", len(y_true))

        grouped = accuracy(y_pred, y_true, self._known_classes, self._total_classes - self._known_classes)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():

                outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]

            # print(predicts.shape)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true) # [N, topk]

    