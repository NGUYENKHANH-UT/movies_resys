import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import DragonGCN
from .config import Config

class MARGO(nn.Module):
    """
    MARGO: Paper-exact implementation following all equations precisely.
    
    Key Implementation Details:
    1. z (reliability) computed from modality-specific ratings BEFORE fusion (Eq. 5-6)
    2. γ (confidence) computed from final fused ratings AFTER fusion (Eq. 7)
    3. Hard threshold g(x) mapping function (Eq. 6)
    4. tanh confidence instead of sigmoid (Eq. 7)
    5. KL divergence for calibration loss (Eq. 8)
    """
    def __init__(self, num_users, num_items, edge_index):
        super(MARGO, self).__init__()
        self.device = Config.device
        self.num_items = num_items
        
        # --- BACKBONE: 2 GNN Branches (Equation 1) ---
        self.v_gcn = DragonGCN(num_users, num_items, Config.feat_dim_v, Config.embed_dim, edge_index, self.device)
        self.t_gcn = DragonGCN(num_users, num_items, Config.feat_dim_t, Config.embed_dim, edge_index, self.device)
        
        # --- MARGO PARAMETERS ---
        # Initialize weights to zeros → softmax(0,0) = (0.5, 0.5)
        init_weights = torch.zeros(num_items, 2)
        self.item_modality_weights = nn.Parameter(init_weights.to(self.device))
        
        self.stage = 1
        self.current_alpha = 0.0
        self.last_loss_dict = {}
        
        # --- Metrics for MLflow Logging ---
        self.last_gamma_mean = 0.0
        self.last_gamma_min = 0.0
        self.last_gamma_max = 0.0
        self.last_gamma_zero = 0.0
        self.last_kl_mean = 0.0
        self.last_weights_v_mean = 0.5
        self.last_weights_t_mean = 0.5

    def g_mapping(self, x):
        """
        Paper Equation 6: Mapping function g()
        g(x) = x,      if x >= 0
        g(x) = -e^6,   if x < 0
        
        This ensures unreliable modality gets reliability ≈ 0 after softmax.
        """
        return torch.where(x >= 0, x, torch.full_like(x, -1e6))

    def compute_modality_reliability(self, pos_score_v, pos_score_t, neg_score_v, neg_score_t):
        """
        Paper Equation 5-6: Modality Reliability Vector
        
        IMPORTANT: This uses modality-specific ratings y^m_ui (BEFORE fusion with weights)
        
        d_uik = [y^v_ui - y^v_uk, y^t_ui - y^t_uk]
        z_uik = softmax(g(d_uik))
        """
        # Step 1: Calculate difference vector (Equation 5)
        diff_v = pos_score_v - neg_score_v
        diff_t = pos_score_t - neg_score_t
        
        # Step 2: Apply mapping function g() (Equation 6)
        z_v_logit = self.g_mapping(diff_v)
        z_t_logit = self.g_mapping(diff_t)
        
        # Step 3: Softmax to get reliability distribution
        z = F.softmax(torch.stack([z_v_logit, z_t_logit], dim=1), dim=1)
        
        return z.detach()

    def compute_confidence(self, pos_score, neg_score):
        """
        Paper Equation 7: Confidence
        
        IMPORTANT: This uses final fused rating y_ui (AFTER fusion with weights)
        
        γ_uik = tanh((y_ui - y_uk) / τ),  if y_ui > y_uk
        γ_uik = 0,                         if y_ui <= y_uk
        """
        score_diff = pos_score - neg_score
        
        # Apply tanh scaling
        gamma = torch.tanh(score_diff / Config.tau)
        
        # Hard threshold at 0
        gamma = torch.where(
            score_diff > 0,
            gamma,
            torch.zeros_like(gamma)
        )
        
        return gamma.detach()

    def compute_calibration_loss(self, z, gamma, w_pos, w_neg):
        """
        Paper Equation 8: Weight Calibration Loss
        
        L_cal = Σ nograd(γ_uik) * KL(nograd(z_uik) || w_i ⊕ w_k)
        
        Using KL Divergence:
        KL(p||q) = Σ p * log(p/q)
        """
        # Paper uses w_i ⊕ w_k (element-wise sum)
        # We normalize to keep it as probability distribution
        w_sum = w_pos + w_neg
        w_sum = F.softmax(w_sum, dim=1)  # Re-normalize
        
        # KL Divergence (Paper Equation 8)
        epsilon = 1e-8
        kl_div = torch.sum(
            z * (torch.log(z + epsilon) - torch.log(w_sum + epsilon)),
            dim=1
        )
        
        # Weighted by confidence
        cal_loss = torch.mean(gamma * kl_div)
        
        return cal_loss, kl_div

    def forward(self, batch_data, feat_v, feat_t):
        """
        Forward pass following paper pipeline exactly.
        
        Correct Order:
        1. Compute modality-specific ratings (Eq. 2)
        2. Compute reliability z from modality-specific ratings (Eq. 5-6) ← BEFORE fusion
        3. Fusion with weights → final rating (Eq. 3)
        4. Compute confidence γ from final rating (Eq. 7) ← AFTER fusion
        5. Calibration loss (Eq. 8)
        """
        u_ids, pos_ids, neg_ids = batch_data
        
        # ====================================================
        # STEP 1: Get Embeddings (Equation 1)
        # ====================================================
        u_v_all, i_v_all = self.v_gcn(feat_v)
        u_t_all, i_t_all = self.t_gcn(feat_t)
        
        u_v, u_t = u_v_all[u_ids], u_t_all[u_ids]
        pos_iv, pos_it = i_v_all[pos_ids], i_t_all[pos_ids]
        neg_iv, neg_it = i_v_all[neg_ids], i_t_all[neg_ids]
        
        # ====================================================
        # STEP 2: Modality-Specific Ratings (Equation 2)
        # These are y^m_ui - BEFORE fusion with weights
        # ====================================================
        pos_score_v = (u_v * pos_iv).sum(dim=1)
        pos_score_t = (u_t * pos_it).sum(dim=1)
        neg_score_v = (u_v * neg_iv).sum(dim=1)
        neg_score_t = (u_t * neg_it).sum(dim=1)
        
        # ====================================================
        # STEP 3: Compute Reliability Vector (Equation 5-6)
        # CRITICAL: Must be computed BEFORE fusion with weights!
        # z uses modality-specific ratings y^m_ui
        # ====================================================
        if self.stage == 2 and self.current_alpha > 0:
            z = self.compute_modality_reliability(
                pos_score_v, pos_score_t,
                neg_score_v, neg_score_t
            )
        else:
            z = None
        
        # ====================================================
        # STEP 4: Get Modality Weights
        # ====================================================
        w_pos = F.softmax(self.item_modality_weights[pos_ids], dim=1)
        w_neg = F.softmax(self.item_modality_weights[neg_ids], dim=1)
        
        # Track modality weights for logging (Stage 2)
        if self.stage == 2:
            with torch.no_grad():
                self.last_weights_v_mean = w_pos[:, 0].mean().item()
                self.last_weights_t_mean = w_pos[:, 1].mean().item()
        
        # ====================================================
        # STEP 5: Score Fusion (Equation 3)
        # This produces y_ui - AFTER fusion with weights
        # ====================================================
        if self.stage == 1:
            # Stage 1: Simple sum (Equation 9)
            pos_score = pos_score_v + pos_score_t
            neg_score = neg_score_v + neg_score_t
        else:
            # Stage 2: Weighted fusion (Equation 3)
            pos_score = w_pos[:, 0] * pos_score_v + w_pos[:, 1] * pos_score_t
            neg_score = w_neg[:, 0] * neg_score_v + w_neg[:, 1] * neg_score_t
        
        # ====================================================
        # STEP 6: BPR Loss (Equation 4)
        # ====================================================
        bpr_loss = F.softplus(neg_score - pos_score).mean()
        
        # ====================================================
        # STEP 7: Regularization (Equation 9-10)
        # ====================================================
        reg_loss = Config.weight_decay * (
            self.v_gcn.preference.pow(2).sum() + 
            self.t_gcn.preference.pow(2).sum()
        ) / 2.0
        
        loss = bpr_loss + reg_loss
        
        self.last_loss_dict = {
            'total': loss.item(),
            'bpr': bpr_loss.item(),
            'reg': reg_loss.item(),
            'cal': 0.0
        }
        
        # ====================================================
        # STEP 8: Calibration Loss (Stage 2, Equation 7-8)
        # ====================================================
        if self.stage == 2 and self.current_alpha > 0 and z is not None:
            # Compute confidence from FINAL fused ratings (Equation 7)
            # γ uses y_ui - AFTER fusion with weights
            gamma = self.compute_confidence(pos_score, neg_score)
            
            # Compute calibration loss (Equation 8)
            cal_loss, kl_div = self.compute_calibration_loss(z, gamma, w_pos, w_neg)
            
            loss = loss + self.current_alpha * cal_loss
            self.last_loss_dict['cal'] = cal_loss.item()
            self.last_loss_dict['total'] = loss.item()
            
            # ====================================================
            # STEP 9: Track Metrics for MLflow Logging
            # ====================================================
            with torch.no_grad():
                # Gamma statistics
                self.last_gamma_mean = gamma.mean().item()
                self.last_gamma_min = gamma.min().item()
                self.last_gamma_max = gamma.max().item()
                self.last_gamma_zero = (gamma == 0).float().mean().item()
                
                # KL divergence mean (raw values before weighting)
                self.last_kl_mean = kl_div.mean().item()
        
        return loss

    def get_final_embeddings(self, feat_v, feat_t):
        """Helper for inference/evaluation"""
        with torch.no_grad():
            u_v, i_v = self.v_gcn(feat_v)
            u_t, i_t = self.t_gcn(feat_t)
            return (u_v, u_t), (i_v, i_t)