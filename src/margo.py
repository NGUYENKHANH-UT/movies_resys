import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import DragonGCN
from .config import Config

class MARGO(nn.Module):
    def __init__(self, num_users, num_items, edge_index):
        super(MARGO, self).__init__()
        self.device = Config.device
        self.num_items = num_items
        
        # --- BACKBONE: 2 GNN Branches ---
        self.v_gcn = DragonGCN(num_users, num_items, Config.feat_dim_v, Config.embed_dim, edge_index, self.device)
        self.t_gcn = DragonGCN(num_users, num_items, Config.feat_dim_t, Config.embed_dim, edge_index, self.device)
        
        # --- MARGO PARAMETERS ---
        # This helps learning converge faster
        init_weights = torch.ones(num_items, 2) * 0.5  # Start with equal weights
        self.item_modality_weights = nn.Parameter(init_weights.to(self.device))
        
        # Stage control
        self.stage = 1
        
        self.current_alpha = 0.0

    def forward(self, batch_data, feat_v, feat_t):
        """
        Forward pass with improved loss computation.
        """
        u_ids, pos_ids, neg_ids = batch_data
        
        # Step 1: Get embeddings from both GCN branches
        u_v_all, i_v_all = self.v_gcn(feat_v)
        u_t_all, i_t_all = self.t_gcn(feat_t)
        
        # Step 2: Lookup embeddings for current batch
        u_v, u_t = u_v_all[u_ids], u_t_all[u_ids]
        pos_iv, pos_it = i_v_all[pos_ids], i_t_all[pos_ids]
        neg_iv, neg_it = i_v_all[neg_ids], i_t_all[neg_ids]
        
        # Step 3: Calculate component scores
        pos_score_v = (u_v * pos_iv).sum(dim=1)
        pos_score_t = (u_t * pos_it).sum(dim=1)
        neg_score_v = (u_v * neg_iv).sum(dim=1)
        neg_score_t = (u_t * neg_it).sum(dim=1)
        
        # Step 4: Score fusion based on current stage
        w_pos = F.softmax(self.item_modality_weights[pos_ids], dim=1)
        w_neg = F.softmax(self.item_modality_weights[neg_ids], dim=1)
        
        if self.stage == 1:
            pos_score = pos_score_v + pos_score_t
            neg_score = neg_score_v + neg_score_t
        else:
            pos_score = w_pos[:, 0] * pos_score_v + w_pos[:, 1] * pos_score_t
            neg_score = w_neg[:, 0] * neg_score_v + w_neg[:, 1] * neg_score_t
        
        # softplus(-x) = log(1 + exp(-x)) is more stable than -log(sigmoid(x))
        bpr_loss = F.softplus(neg_score - pos_score).mean()
        
        # Step 6: Regularization
        reg_loss = Config.weight_decay * (
            self.v_gcn.preference.pow(2).sum() + 
            self.t_gcn.preference.pow(2).sum()
        ) / 2.0
        
        loss = bpr_loss + reg_loss
        
        # Dictionary to return individual components for logging
        loss_dict = {
            'total': loss.item(),
            'bpr': bpr_loss.item(),
            'reg': reg_loss.item(),
            'cal': 0.0
        }
        
        # Step 7: Calibration loss (only in Stage 2)
        if self.stage == 2 and self.current_alpha > 0:
            # Use smooth mapping instead of hard threshold
            diff_v = pos_score_v - neg_score_v
            diff_t = pos_score_t - neg_score_t
            
            # Smooth soft-thresholding: ReLU instead of hard -1e9
            z_v_logit = F.relu(diff_v) + 1e-6  # Add small epsilon for stability
            z_t_logit = F.relu(diff_t) + 1e-6
            
            # Softmax to get reliability distribution
            z = F.softmax(torch.stack([z_v_logit, z_t_logit], dim=1), dim=1).detach()
            
            # Clip the difference to prevent extreme values
            score_diff = torch.clamp(pos_score - neg_score, min=-5.0, max=5.0)
            gamma = torch.sigmoid(score_diff / Config.tau).detach()  # Use sigmoid instead of tanh
            
            # Clip gamma to avoid extreme confidence (0.1 to 0.9)
            gamma = torch.clamp(gamma, min=0.1, max=0.9)
            
            # Average weights for positive and negative items
            w_avg = (w_pos + w_neg) / 2.0
            
            # Add epsilon to prevent log(0)
            epsilon = 1e-8
            kl_div = torch.sum(
                z * (torch.log(z + epsilon) - torch.log(w_avg + epsilon)), 
                dim=1
            )
            
            # Clip KL divergence to prevent explosion
            kl_div = torch.clamp(kl_div, min=0.0, max=10.0)
            
            # Weighted calibration loss
            cal_loss = torch.mean(gamma * kl_div)
            
            loss = loss + self.current_alpha * cal_loss
            loss_dict['cal'] = cal_loss.item()
            loss_dict['total'] = loss.item()
        
        # Store for logging
        self.last_loss_dict = loss_dict
        
        return loss

    def get_final_embeddings(self, feat_v, feat_t):
        """
        Helper for Inference/Evaluation
        """
        with torch.no_grad():
            u_v, i_v = self.v_gcn(feat_v)
            u_t, i_t = self.t_gcn(feat_t)
            return (u_v, u_t), (i_v, i_t)