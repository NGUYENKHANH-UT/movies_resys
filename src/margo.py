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
        init_weights = torch.ones(num_items, 2) * 0.5 
        self.item_modality_weights = nn.Parameter(init_weights.to(self.device))
        
        self.stage = 1
        self.current_alpha = 0.0

    def forward(self, batch_data, feat_v, feat_t):
        u_ids, pos_ids, neg_ids = batch_data
        
        # Step 1: Get embeddings
        u_v_all, i_v_all = self.v_gcn(feat_v)
        u_t_all, i_t_all = self.t_gcn(feat_t)
        
        # Step 2: Lookup embeddings
        u_v, u_t = u_v_all[u_ids], u_t_all[u_ids]
        pos_iv, pos_it = i_v_all[pos_ids], i_t_all[pos_ids]
        neg_iv, neg_it = i_v_all[neg_ids], i_t_all[neg_ids]
        
        # Step 3: Calculate component scores
        pos_score_v = (u_v * pos_iv).sum(dim=1)
        pos_score_t = (u_t * pos_it).sum(dim=1)
        neg_score_v = (u_v * neg_iv).sum(dim=1)
        neg_score_t = (u_t * neg_it).sum(dim=1)
        
        # Step 4: Score fusion
        w_pos = F.softmax(self.item_modality_weights[pos_ids], dim=1)
        w_neg = F.softmax(self.item_modality_weights[neg_ids], dim=1)
        
        if self.stage == 1:
            pos_score = pos_score_v + pos_score_t
            neg_score = neg_score_v + neg_score_t
        else:
            pos_score = w_pos[:, 0] * pos_score_v + w_pos[:, 1] * pos_score_t
            neg_score = w_neg[:, 0] * neg_score_v + w_neg[:, 1] * neg_score_t
        
        # Step 5: BPR Loss
        bpr_loss = F.softplus(neg_score - pos_score).mean()
        
        # Step 6: Regularization
        reg_loss = Config.weight_decay * (
            self.v_gcn.preference.pow(2).sum() + 
            self.t_gcn.preference.pow(2).sum()
        ) / 2.0
        
        loss = bpr_loss + reg_loss
        
        loss_dict = {
            'total': loss.item(),
            'bpr': bpr_loss.item(),
            'reg': reg_loss.item(),
            'cal': 0.0
        }
        
        # Step 7: Calibration loss (Stage 2)
        if self.stage == 2 and self.current_alpha > 0:
            diff_v = pos_score_v - neg_score_v
            diff_t = pos_score_t - neg_score_t
            
            # --- FIX 1: Soft mapping thay vì hard threshold ---
            # Thay vì dùng threshold cứng, dùng ReLU + small positive bias
            # Cho phép gradient flow tốt hơn và tránh zero gradient
            z_v_logit = F.relu(diff_v) + 0.1  # Bias nhỏ để tránh zero
            z_t_logit = F.relu(diff_t) + 0.1
            
            # Softmax để tạo reliability vector
            # Khi cả 2 đều xấu: [0.1, 0.1] → softmax → [0.5, 0.5] (fair)
            # Khi v tốt, t xấu: [2.1, 0.1] → softmax → [0.89, 0.11] (v reliable)
            z = F.softmax(torch.stack([z_v_logit, z_t_logit], dim=1), dim=1).detach()
            
            # --- FIX 2: Confidence mechanism cải thiện ---
            # Paper dùng sigmoid, nhưng ở early epochs score_diff còn nhỏ
            # → gamma ≈ 0.5 → supervision quá mạnh cho unreliable predictions
            # Solution: Chỉ supervise khi score_diff > threshold
            score_diff = pos_score - neg_score  # Không clamp quá sớm
            
            # Tính raw confidence
            gamma_raw = torch.sigmoid(score_diff / Config.tau)
            
            # CRITICAL: Chỉ supervise predictions có confidence cao (> 0.6)
            # Với tau=1.0, điều này tương đương score_diff > 0.4
            confidence_threshold = 0.6
            gamma = torch.where(
                gamma_raw > confidence_threshold,
                gamma_raw,
                torch.zeros_like(gamma_raw)
            ).detach()
            
            # Average weights của pos và neg items
            w_avg = (w_pos + w_neg) / 2.0
            
            # --- FIX 3: KL Divergence với clipping hợp lý ---
            epsilon = 1e-8
            kl_div = torch.sum(
                z * (torch.log(z + epsilon) - torch.log(w_avg + epsilon)), 
                dim=1
            )
            
            # Clip KL nhưng không quá aggressive
            kl_div = torch.clamp(kl_div, min=0.0, max=5.0)  # Giảm từ 10.0 xuống 5.0
            
            # --- FIX 4: Chỉ tính loss trên samples có confidence > 0 ---
            # Tránh supervise những triplets không confident
            valid_mask = (gamma > 0).float()
            
            if valid_mask.sum() > 0:  # Có ít nhất 1 confident sample
                cal_loss = torch.sum(gamma * kl_div) / (valid_mask.sum() + epsilon)
            else:
                cal_loss = torch.tensor(0.0).to(self.device)
            
            loss = loss + self.current_alpha * cal_loss
            loss_dict['cal'] = cal_loss.item()
            loss_dict['total'] = loss.item()
            
            # Logging để debug
            self.last_gamma_mean = gamma.mean().item()
            self.last_valid_ratio = valid_mask.mean().item()
        
        self.last_loss_dict = loss_dict
        return loss

    def get_final_embeddings(self, feat_v, feat_t):
        with torch.no_grad():
            u_v, i_v = self.v_gcn(feat_v)
            u_t, i_t = self.t_gcn(feat_t)
            return (u_v, u_t), (i_v, i_t)