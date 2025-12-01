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
        Calculate Loss for a training batch.
        """
        u_ids, pos_ids, neg_ids = batch_data
        
        # 1. Get Embeddings from Backbone
        # Note: feat_v, feat_t are tensors loaded from Milvus in Dataset
        u_v_all, i_v_all = self.v_gcn(feat_v)
        u_t_all, i_t_all = self.t_gcn(feat_t)
        
        # 2. Lookup Embeddings for current Batch
        u_v, u_t = u_v_all[u_ids], u_t_all[u_ids]
        
        pos_iv, pos_it = i_v_all[pos_ids], i_t_all[pos_ids]
        neg_iv, neg_it = i_v_all[neg_ids], i_t_all[neg_ids]
        
        # 3. Calculate Component Scores
        # Dot product: (Batch, Dim) * (Batch, Dim) -> (Batch, 1)
        pos_score_v = (u_v * pos_iv).sum(dim=1)
        pos_score_t = (u_t * pos_it).sum(dim=1)
        
        neg_score_v = (u_v * neg_iv).sum(dim=1)
        neg_score_t = (u_t * neg_it).sum(dim=1)
        
        # 4. Score Fusion
        # Get Softmax-normalized weights
        w_pos = F.softmax(self.item_modality_weights[pos_ids], dim=1)
        w_neg = F.softmax(self.item_modality_weights[neg_ids], dim=1)
        
        if self.stage == 1:
            # Stage 1: Equal summation (equivalent to w=0.5)
            pos_score = pos_score_v + pos_score_t
            neg_score = neg_score_v + neg_score_t
        else:
            # Stage 2: MARGO Weighted Fusion
            pos_score = w_pos[:, 0] * pos_score_v + w_pos[:, 1] * pos_score_t
            neg_score = w_neg[:, 0] * neg_score_v + w_neg[:, 1] * neg_score_t
            
        # 5. Calculate BPR Loss (Main Loss)
        # Softplus(x) = log(1 + exp(x)) is equivalent to -log_sigmoid
        # Loss = -ln(sigmoid(pos - neg))
        bpr_loss = F.softplus(-(pos_score - neg_score)).mean()
        
        # Regularization (L2 Norm on user preference to prevent overfitting)
        reg_loss = Config.weight_decay * (
            self.v_gcn.preference.norm(2) + self.t_gcn.preference.norm(2)
        )
        
        loss = bpr_loss + reg_loss
        
        # 6. Calculate Calibration Loss (Only active in Stage 2)
        if self.stage == 2:
            # A. Create Reliability Vector 'z' (Self-supervised Label)
            diff_v = pos_score_v - neg_score_v
            diff_t = pos_score_t - neg_score_t
            
            # g(x) function: keep positive, map negative to -infinity
            min_val = torch.tensor(-1e9).to(self.device)
            z_v = torch.where(diff_v >= 0, diff_v, min_val)
            z_t = torch.where(diff_t >= 0, diff_t, min_val)
            
            # Apply softmax to get probability distribution z
            z = F.softmax(torch.stack([z_v, z_t], dim=1), dim=1).detach()
            
            # B. Confidence (Gamma)
            gamma = torch.tanh((pos_score - neg_score) / Config.tau).detach()
            gamma = torch.clamp(gamma, min=0) # Only consider correct predictions
            
            # C. KL Divergence: sum(z * log(z/w))
            # Average weight of positive and negative items
            w_avg = (w_pos + w_neg) / 2
            
            # Add epsilon to avoid log(0)
            log_w = torch.log(w_avg + 1e-10)
            log_z = torch.log(z + 1e-10)
            
            # KL sum over modalities (dim=1)
            kl_div = torch.sum(z * (log_z - log_w), dim=1)
            
            # Final Calibration Loss = Mean(Gamma * KL)
            cal_loss = torch.mean(gamma * kl_div)
            
            loss += Config.alpha * cal_loss
            
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