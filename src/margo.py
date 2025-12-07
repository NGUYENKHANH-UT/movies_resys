import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import DragonGCN
from .config import Config

class MARGO(nn.Module):
    """
    MARGO: Modality Reliability Guided Multimodal Recommendation
    Paper-exact implementation following equations in the paper.
    """
    def __init__(self, num_users, num_items, edge_index):
        super(MARGO, self).__init__()
        self.device = Config.device
        self.num_items = num_items
        
        # --- BACKBONE: 2 GNN Branches (Equation 1) ---
        self.v_gcn = DragonGCN(num_users, num_items, Config.feat_dim_v, Config.embed_dim, edge_index, self.device)
        self.t_gcn = DragonGCN(num_users, num_items, Config.feat_dim_t, Config.embed_dim, edge_index, self.device)
        
        # --- MARGO PARAMETERS ---
        # Initialize with equal weights (0.5, 0.5) before softmax
        # After softmax: each item has [w_v, w_t] that sum to 1
        init_weights = torch.zeros(num_items, 2)  # Will become (0.5, 0.5) after softmax
        self.item_modality_weights = nn.Parameter(init_weights.to(self.device))
        
        # Stage control
        self.stage = 1  # 1 = pre-training, 2 = fine-tuning
        
        # Current alpha (will be updated by scheduler)
        self.current_alpha = 0.0
        
        # For logging
        self.last_loss_dict = {}

    def g_mapping(self, x):
        """
        Paper Equation 6: Mapping function g()
        g(x) = x,      if x >= 0
        g(x) = -e^6,   if x < 0
        
        This enforces that unreliable modality gets reliability = 0 after softmax.
        """
        # -e^6 ≈ -403.43 (very large negative number)
        return torch.where(x >= 0, x, torch.full_like(x, -1e6))

    def compute_modality_reliability(self, pos_score_v, pos_score_t, neg_score_v, neg_score_t):
        """
        Paper Equation 5-6: Modality Reliability Vector
        
        d_uik = [y_ui^v - y_uk^v, y_ui^t - y_uk^t]
        z_uik = softmax(g(d_uik))
        
        Returns:
            z: [batch_size, 2] - Reliability distribution (sums to 1)
        """
        # Step 1: Calculate difference vector (Equation 5)
        diff_v = pos_score_v - neg_score_v
        diff_t = pos_score_t - neg_score_t
        
        # Step 2: Apply mapping function g() (Equation 6)
        z_v_logit = self.g_mapping(diff_v)
        z_t_logit = self.g_mapping(diff_t)
        
        # Step 3: Softmax to get reliability distribution
        # Shape: [batch_size, 2]
        z = F.softmax(torch.stack([z_v_logit, z_t_logit], dim=1), dim=1)
        
        # Detach to stop gradient (nograd in paper)
        return z.detach()

    def compute_confidence(self, pos_score, neg_score):
        """
        Paper Equation 7: Confidence for Modality Reliability Vector
        
        γ_uik = tanh((y_ui - y_uk) / τ),  if y_ui > y_uk
        γ_uik = 0,                         if y_ui <= y_uk
        
        Returns:
            gamma: [batch_size] - Confidence values in [0, 1)
        """
        score_diff = pos_score - neg_score
        
        # Apply tanh with temperature scaling
        gamma = torch.tanh(score_diff / Config.tau)
        
        # Hard threshold: set to 0 if y_ui <= y_uk
        gamma = torch.where(
            score_diff > 0,
            gamma,
            torch.zeros_like(gamma)
        )
        
        # Detach to stop gradient (nograd in paper)
        return gamma.detach()

    def compute_calibration_loss(self, z, gamma, w_pos, w_neg):
        """
        Paper Equation 8: Weight Calibration Loss
        
        L_cal = Σ nograd(γ_uik) * KL(nograd(z_uik) || w_i ⊕ w_k)
        
        where ⊕ is element-wise summation
        
        Args:
            z: [batch_size, 2] - Modality reliability (detached)
            gamma: [batch_size] - Confidence (detached)
            w_pos: [batch_size, 2] - Weights of positive items
            w_neg: [batch_size, 2] - Weights of negative items
        
        Returns:
            cal_loss: scalar
        """
        # Paper uses w_i ⊕ w_k (element-wise sum)
        # However, this creates a problem: sum may exceed 1
        # We interpret this as: "joint weight distribution of both items"
        # Solution: Normalize the sum to keep it as valid probability
        w_sum = w_pos + w_neg
        w_sum = F.softmax(w_sum, dim=1)  # Re-normalize to sum to 1
        
        # KL Divergence: KL(z || w_sum) = Σ z * log(z / w_sum)
        epsilon = 1e-8
        kl_div = torch.sum(
            z * (torch.log(z + epsilon) - torch.log(w_sum + epsilon)),
            dim=1
        )
        
        # Weighted by confidence (Equation 8)
        cal_loss = torch.mean(gamma * kl_div)
        
        return cal_loss

    def forward(self, batch_data, feat_v, feat_t):
        """
        Forward pass following MARGO paper pipeline.
        
        Stage 1: Simple sum fusion (Equation 9)
        Stage 2: Weighted fusion + calibration (Equation 10)
        """
        u_ids, pos_ids, neg_ids = batch_data
        
        # ====================================================
        # STEP 1: Get Embeddings from GCN (Equation 1-2)
        # ====================================================
        u_v_all, i_v_all = self.v_gcn(feat_v)
        u_t_all, i_t_all = self.t_gcn(feat_t)
        
        # Lookup embeddings for current batch
        u_v, u_t = u_v_all[u_ids], u_t_all[u_ids]
        pos_iv, pos_it = i_v_all[pos_ids], i_t_all[pos_ids]
        neg_iv, neg_it = i_v_all[neg_ids], i_t_all[neg_ids]
        
        # ====================================================
        # STEP 2: Calculate Modality-Specific Ratings (Equation 2)
        # ====================================================
        pos_score_v = (u_v * pos_iv).sum(dim=1)  # y_ui^v
        pos_score_t = (u_t * pos_it).sum(dim=1)  # y_ui^t
        neg_score_v = (u_v * neg_iv).sum(dim=1)  # y_uk^v
        neg_score_t = (u_t * neg_it).sum(dim=1)  # y_uk^t
        
        # ====================================================
        # STEP 3: Fuse Scores Based on Stage (Equation 3)
        # ====================================================
        # Get softmax-normalized weights
        w_pos = F.softmax(self.item_modality_weights[pos_ids], dim=1)  # [batch, 2]
        w_neg = F.softmax(self.item_modality_weights[neg_ids], dim=1)  # [batch, 2]
        
        if self.stage == 1:
            # Stage 1: Simple summation (Equation 9)
            pos_score = pos_score_v + pos_score_t
            neg_score = neg_score_v + neg_score_t
        else:
            # Stage 2: Weighted fusion (Equation 3)
            pos_score = w_pos[:, 0] * pos_score_v + w_pos[:, 1] * pos_score_t
            neg_score = w_neg[:, 0] * neg_score_v + w_neg[:, 1] * neg_score_t
        
        # ====================================================
        # STEP 4: BPR Loss (Equation 4)
        # ====================================================
        # Paper: L_rec = -Σ log(σ(y_ui - y_uk))
        # Equivalent stable form: softplus(-(y_ui - y_uk))
        bpr_loss = F.softplus(neg_score - pos_score).mean()
        
        # ====================================================
        # STEP 5: Regularization (Equation 9-10)
        # ====================================================
        reg_loss = Config.weight_decay * (
            self.v_gcn.preference.pow(2).sum() + 
            self.t_gcn.preference.pow(2).sum()
        ) / 2.0
        
        # Total loss (Stage 1)
        loss = bpr_loss + reg_loss
        
        # Loss dictionary for logging
        self.last_loss_dict = {
            'total': loss.item(),
            'bpr': bpr_loss.item(),
            'reg': reg_loss.item(),
            'cal': 0.0
        }
        
        # ====================================================
        # STEP 6: Calibration Loss (Stage 2 Only, Equation 8)
        # ====================================================
        if self.stage == 2 and self.current_alpha > 0:
            # Compute modality reliability vector (Equation 5-6)
            z = self.compute_modality_reliability(
                pos_score_v, pos_score_t,
                neg_score_v, neg_score_t
            )
            
            # Compute confidence (Equation 7)
            gamma = self.compute_confidence(pos_score, neg_score)
            
            # Compute calibration loss (Equation 8)
            cal_loss = self.compute_calibration_loss(z, gamma, w_pos, w_neg)
            
            # Add to total loss (Equation 10)
            loss = loss + self.current_alpha * cal_loss
            
            # Update loss dict
            self.last_loss_dict['cal'] = cal_loss.item()
            self.last_loss_dict['total'] = loss.item()
        
        return loss

    def get_final_embeddings(self, feat_v, feat_t):
        """
        Helper for Inference/Evaluation.
        Returns embeddings from both modalities.
        """
        with torch.no_grad():
            u_v, i_v = self.v_gcn(feat_v)
            u_t, i_t = self.t_gcn(feat_t)
            return (u_v, u_t), (i_v, i_t)