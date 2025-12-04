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
            
            # --- FIX 1: Đúng hàm g(x) từ paper ---
            # Paper: g(x) = x if x >= 0, else g(x) = -e^6 ≈ -403
            # NHƯNG: -403 quá cực đoan, làm softmax bất ổn!
            # Thay bằng -10 (vẫn đảm bảo unreliable → 0 sau softmax)
            neg_penalty = -10.0
            
            z_v_logit = torch.where(diff_v > 0, diff_v, torch.full_like(diff_v, neg_penalty))
            z_t_logit = torch.where(diff_t > 0, diff_t, torch.full_like(diff_t, neg_penalty))
            
            # Softmax để tạo reliability vector
            z = F.softmax(torch.stack([z_v_logit, z_t_logit], dim=1), dim=1).detach()
            
            # --- FIX 2: Confidence với tanh ---
            score_diff = pos_score - neg_score
            
            # Dùng tanh như paper
            gamma = torch.tanh(score_diff / Config.tau)
            
            # Set γ = 0 khi prediction sai (neg_score >= pos_score)
            gamma = torch.where(
                score_diff > 0,
                gamma,
                torch.zeros_like(gamma)
            ).detach()
            
            # --- FIX QUAN TRỌNG 3: JS Divergence ĐƠN GIẢN và ỔN ĐỊNH ---
            # Đừng dùng F.kl_div() vì nó không ổn định với reduction='none'
            # Thay bằng công thức tính JS Divergence trực tiếp
            
            # Average weights như paper
            w_avg = (w_pos + w_neg) / 2.0
            
            # Tính JS Divergence một cách ổn định
            # JS(P||Q) = 0.5 * [KL(P||M) + KL(Q||M)], M = (P+Q)/2
            # KL(P||M) = Σ P * log(P/M)
            
            epsilon = 1e-10
            z_safe = z + epsilon
            w_avg_safe = w_avg + epsilon
            
            # Tính M
            M = 0.5 * (z_safe + w_avg_safe)
            
            # Tính KL divergences
            kl_z = (z_safe * (torch.log(z_safe) - torch.log(M))).sum(dim=1)
            kl_w = (w_avg_safe * (torch.log(w_avg_safe) - torch.log(M))).sum(dim=1)
            
            # JS Divergence
            js_div = 0.5 * (kl_z + kl_w)
            
            # Clamp để ổn định (JS divergence luôn ≤ log(2) ≈ 0.693)
            js_div = torch.clamp(js_div, min=0.0, max=1.0)
            
            # --- FIX 4: Cal loss với normalization an toàn ---
            # Chỉ tính cal loss khi có ít nhất một gamma > 0
            valid_gamma_mask = gamma > 0
            if valid_gamma_mask.any():
                # Chỉ lấy các samples có gamma > 0
                valid_gamma = gamma[valid_gamma_mask]
                valid_js = js_div[valid_gamma_mask]
                
                # Weighted average
                cal_loss = torch.sum(valid_gamma * valid_js) / (torch.sum(valid_gamma) + epsilon)
            else:
                # Không có sample nào valid
                cal_loss = torch.tensor(0.0).to(self.device)
            
            loss = loss + self.current_alpha * cal_loss
            loss_dict['cal'] = cal_loss.item()
            loss_dict['total'] = loss.item()
            
            # Logging để debug
            self.last_gamma_mean = gamma.mean().item()
            self.last_gamma_min = gamma.min().item()
            self.last_gamma_max = gamma.max().item()
            self.last_gamma_zero = (gamma == 0).float().mean().item()
            self.last_kl_mean = js_div.mean().item()
            
            # Logging weights stats
            self.last_weights_v_mean = w_pos[:, 0].mean().item()
            self.last_weights_t_mean = w_pos[:, 1].mean().item()
        
        self.last_loss_dict = loss_dict
        return loss

    def get_final_embeddings(self, feat_v, feat_t):
        with torch.no_grad():
            u_v, i_v = self.v_gcn(feat_v)
            u_t, i_t = self.t_gcn(feat_t)
            return (u_v, u_t), (i_v, i_t)