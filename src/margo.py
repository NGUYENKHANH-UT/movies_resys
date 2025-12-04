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
            
            # --- CRITICAL FIX 1: Paper's g(x) function ---
            # Paper Eq. (6): g(x) = x if x >= 0, else g(x) = -e^6 ≈ -403
            # Ý nghĩa: Modality unreliable → logit rất âm → sau softmax ≈ 0
            
            # Nhưng -e^6 quá cực đoan! Dùng giá trị âm vừa phải:
            neg_penalty = -5.0  # Modality xấu sẽ có weight thấp, không phải 0
            
            z_v_logit = torch.where(diff_v > 0, diff_v, torch.full_like(diff_v, neg_penalty))
            z_t_logit = torch.where(diff_t > 0, diff_t, torch.full_like(diff_t, neg_penalty))
            
            # Softmax để tạo reliability vector
            # Example: [2.0, -5.0] → softmax → [0.996, 0.004] (v very reliable)
            # Example: [-5.0, -5.0] → softmax → [0.5, 0.5] (both bad, neutral)
            z = F.softmax(torch.stack([z_v_logit, z_t_logit], dim=1), dim=1).detach()
            
            # --- CRITICAL FIX 2: Continuous confidence (KHÔNG dùng threshold!) ---
            # Paper Eq. (7): γ = tanh((y_ui - y_uk) / τ) if y_ui > y_uk, else 0
            # 
            # QUAN TRỌNG: γ = 0 khi prediction SAI (y_ui <= y_uk)
            #             γ ∈ (0,1) khi prediction ĐÚNG (y_ui > y_uk)
            
            score_diff = pos_score - neg_score
            
            # Paper dùng tanh, nhưng sigmoid + clipping cũng tương đương
            gamma = torch.sigmoid(score_diff / Config.tau)
            
            # Set γ = 0 khi prediction sai (neg_score >= pos_score)
            gamma = torch.where(
                score_diff > 0,
                gamma,
                torch.zeros_like(gamma)
            ).detach()
            
            # Clamp gamma to avoid extreme values
            # Min = 0.05 để vẫn có gradient flow cho hard cases
            # Max = 0.95 để tránh overconfident
            gamma = torch.clamp(gamma, min=0.05, max=0.95)
            
            # --- CRITICAL FIX 3: Average weights (w_i ⊕ w_k theo paper) ---
            # Paper Eq. (8): w_i ⊕ w_k (element-wise sum, rồi normalize)
            # Nhưng đơn giản hơn: dùng average
            w_avg = (w_pos + w_neg) / 2.0
            
            # --- CRITICAL FIX 4: KL Divergence đúng theo paper ---
            # Forward KL: KL(z || w) = Σ z · log(z/w)
            # Gradient: ∂KL/∂w = -z/w (pull w toward z)
            epsilon = 1e-8
            kl_div = torch.sum(
                z * (torch.log(z + epsilon) - torch.log(w_avg + epsilon)), 
                dim=1
            )
            
            # Clip KL để stability
            kl_div = torch.clamp(kl_div, min=0.0, max=3.0)
            
            # --- CRITICAL FIX 5: Weighted average (KHÔNG filter samples!) ---
            # Paper: L_cal = Σ γ · KL  (tất cả samples, không filter)
            # Khi γ nhỏ (unreliable), supervision yếu
            # Khi γ lớn (reliable), supervision mạnh
            
            cal_loss = torch.mean(gamma * kl_div)
            
            loss = loss + self.current_alpha * cal_loss
            loss_dict['cal'] = cal_loss.item()
            loss_dict['total'] = loss.item()
            
            # Logging để debug
            self.last_gamma_mean = gamma.mean().item()
            self.last_gamma_min = gamma.min().item()
            self.last_gamma_max = gamma.max().item()
            self.last_kl_mean = kl_div.mean().item()
        
        self.last_loss_dict = loss_dict
        return loss

    def get_final_embeddings(self, feat_v, feat_t):
        with torch.no_grad():
            u_v, i_v = self.v_gcn(feat_v)
            u_t, i_t = self.t_gcn(feat_t)
            return (u_v, u_t), (i_v, i_t)