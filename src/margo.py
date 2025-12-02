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
        # Visual Branch
        self.v_gcn = DragonGCN(num_users, num_items, Config.feat_dim_v, Config.embed_dim, edge_index, self.device)
        # Textual Branch
        self.t_gcn = DragonGCN(num_users, num_items, Config.feat_dim_t, Config.embed_dim, edge_index, self.device)
        
        # --- MARGO PARAMETERS ---
        # Reliability Weights W [N_items, 2]. Column 0: Visual, Column 1: Text
        # Initialize randomly
        self.item_modality_weights = nn.Parameter(torch.randn(num_items, 2).to(self.device))
        
        # Stage control variable
        self.stage = 1

    def forward(self, batch_data, feat_v, feat_t):
        """
        Forward pass to calculate loss for a training batch.
        
        CRITICAL: This function computes embeddings WITH gradients enabled,
        ensuring proper backpropagation through GCN layers.
        
        Args:
            batch_data: Tuple of (user_ids, pos_item_ids, neg_item_ids)
            feat_v: Visual features [num_items, feat_dim_v]
            feat_t: Textual features [num_items, feat_dim_t]
        
        Returns:
            loss: Total loss (BPR + regularization + calibration if stage 2)
        """
        u_ids, pos_ids, neg_ids = batch_data
        
        # Step 1: Get embeddings from both GCN branches WITH GRADIENTS
        # This is critical - embeddings must be computed in forward pass
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
            # Stage 1: Simple sum (equal weights)
            pos_score = pos_score_v + pos_score_t
            neg_score = neg_score_v + neg_score_t
        else:
            # Stage 2: Weighted fusion
            pos_score = w_pos[:, 0] * pos_score_v + w_pos[:, 1] * pos_score_t
            neg_score = w_neg[:, 0] * neg_score_v + w_neg[:, 1] * neg_score_t
        
        # Step 5: BPR Loss - CORRECT FORMULA
        # We want pos_score > neg_score, so we maximize log(sigmoid(pos - neg))
        # Equivalent to minimizing -log(sigmoid(pos - neg))
        bpr_loss = -F.logsigmoid(pos_score - neg_score).mean()
        
        # Step 6: Regularization (L2 on user preferences only)
        reg_loss = Config.weight_decay * (
            self.v_gcn.preference.pow(2).sum() + 
            self.t_gcn.preference.pow(2).sum()
        ) / 2.0
        
        loss = bpr_loss + reg_loss
        
        # Step 7: Calibration loss (only in Stage 2)
        if self.stage == 2:
            # Calculate modality reliability vector z
            diff_v = pos_score_v - neg_score_v
            diff_t = pos_score_t - neg_score_t
            
            # Map negative differences to very small values
            min_val = torch.tensor(-1e9, device=self.device)
            z_v = torch.where(diff_v >= 0, diff_v, min_val)
            z_t = torch.where(diff_t >= 0, diff_t, min_val)
            
            # Softmax to get reliability distribution
            z = F.softmax(torch.stack([z_v, z_t], dim=1), dim=1).detach()
            
            # Calculate confidence gamma
            gamma = torch.tanh((pos_score - neg_score) / Config.tau).detach()
            gamma = torch.clamp(gamma, min=0.0)
            
            # Average weights for positive and negative items
            w_avg = (w_pos + w_neg) / 2.0
            
            # KL divergence between z and w_avg
            kl_div = torch.sum(
                z * (torch.log(z + 1e-10) - torch.log(w_avg + 1e-10)), 
                dim=1
            )
            
            # Weighted calibration loss
            cal_loss = torch.mean(gamma * kl_div)
            
            loss = loss + Config.alpha * cal_loss
        
        return loss

    def get_final_embeddings(self, feat_v, feat_t):
        """
        Helper for Inference/Evaluation: Returns processed final embeddings
        """
        with torch.no_grad():
            u_v, i_v = self.v_gcn(feat_v)
            u_t, i_t = self.t_gcn(feat_t)
            
            # Return tuple for Evaluator to handle weighted fusion
            return (u_v, u_t), (i_v, i_t)