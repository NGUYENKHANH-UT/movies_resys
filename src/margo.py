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
        # Khởi tạo weights với nhiễu nhỏ để break symmetry
        # Ban đầu là ~0.5, sau đó sẽ tự học lệch đi tùy độ tin cậy
        init_weights = torch.ones(num_items, 2) * 0.5 + torch.randn(num_items, 2) * 0.1
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
            # Stage 1: Cộng gộp đơn giản, không dùng weights
            pos_score = pos_score_v + pos_score_t
            neg_score = neg_score_v + neg_score_t
        else:
            # Stage 2: Dùng trọng số đã học để fuse
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
        
        # Step 7: Calibration loss (Stage 2 ONLY)
        if self.stage == 2 and self.current_alpha > 0:
            diff_v = pos_score_v - neg_score_v
            diff_t = pos_score_t - neg_score_t
            
            # --- Reliability Vector Calculation ---
            # 1. Chuẩn hóa differences
            diff_v_norm = diff_v / (torch.std(diff_v) + 1e-8)
            diff_t_norm = diff_t / (torch.std(diff_t) + 1e-8)
            
            # 2. Hàm g(x): giảm ảnh hưởng của unreliable modality
            neg_scale = 0.2
            z_v_logit = torch.where(diff_v_norm > 0, diff_v_norm, diff_v_norm * neg_scale)
            z_t_logit = torch.where(diff_t_norm > 0, diff_t_norm, diff_t_norm * neg_scale)
            
            # 3. Softmax để ra target distribution z
            temperature = 0.7
            z = F.softmax(torch.stack([z_v_logit, z_t_logit], dim=1) / temperature, dim=1).detach()
            
            # --- Gamma (Confidence) Calculation ---
            score_diff = pos_score - neg_score
            # Sigmoid shifted: score_diff > 1.0 thì gamma mới > 0.5
            gamma = torch.sigmoid((score_diff - 1.0) / Config.tau)
            # Chỉ tin tưởng những sample mà model dự đoán đúng (pos > neg)
            gamma = torch.where(score_diff > 0, gamma, torch.zeros_like(gamma)).detach()
            gamma = torch.clamp(gamma, min=0.1, max=0.9)
            
            # --- JS Divergence Calculation ---
            w_avg = (w_pos + w_neg) / 2.0
            epsilon = 1e-10
            z_safe = z + epsilon
            w_safe = w_avg + epsilon
            
            M = 0.5 * (z_safe + w_safe)
            kl_z = (z_safe * (torch.log(z_safe) - torch.log(M))).sum(dim=1)
            kl_w = (w_safe * (torch.log(w_safe) - torch.log(M))).sum(dim=1)
            js_div = 0.5 * (kl_z + kl_w)
            js_div = torch.clamp(js_div, min=0.0, max=1.0)
            
            # --- FIX QUAN TRỌNG: BỎ ENTROPY REGULARIZATION ---
            # Phần này trước đây ép weights về 0.5, giờ bỏ đi để weights tự do phân cực
            # weight_entropy = - (w_avg * torch.log(w_avg + epsilon)).sum(dim=1).mean()
            # entropy_loss = 0.005 * weight_entropy
            
            # --- Final Cal Loss ---
            valid_gamma_mask = gamma > 0.1
            if valid_gamma_mask.any():
                valid_gamma = gamma[valid_gamma_mask]
                valid_js = js_div[valid_gamma_mask]
                
                # Loss = Trọng số Gamma * Độ lệch JS
                cal_loss = torch.sum(valid_gamma * valid_js) / (torch.sum(valid_gamma) + epsilon)
                
                # cal_loss = cal_loss + entropy_loss  <-- Đã bỏ
            else:
                cal_loss = torch.tensor(0.0).to(self.device)
            
            loss = loss + self.current_alpha * cal_loss
            loss_dict['cal'] = cal_loss.item()
            loss_dict['total'] = loss.item()
            
            # --- Logging Stats ---
            self.last_gamma_mean = gamma.mean().item()
            self.last_gamma_min = gamma.min().item()
            self.last_gamma_max = gamma.max().item()
            self.last_gamma_zero = (gamma < 0.1).float().mean().item()
            self.last_kl_mean = js_div.mean().item()
            
            # Quan trọng: Theo dõi xem weights có lệch khỏi 0.5 không
            self.last_weights_v_mean = w_pos[:, 0].mean().item()
            self.last_weights_t_mean = w_pos[:, 1].mean().item()
        
        self.last_loss_dict = loss_dict
        return loss

    def get_final_embeddings(self, feat_v, feat_t):
        with torch.no_grad():
            u_v, i_v = self.v_gcn(feat_v)
            u_t, i_t = self.t_gcn(feat_t)
            return (u_v, u_t), (i_v, i_t)