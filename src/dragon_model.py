import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbone import DragonGCN
from .config import Config

class DRAGON(nn.Module):
    """
    DRAGON Model implementation aligned with the official paper code.
    """
    def __init__(self, num_users, num_items, edge_index, user_graph_dict, feat_v, feat_t):
        super(DRAGON, self).__init__()
        self.device = Config.device
        self.num_users = num_users
        self.num_items = num_items
        
        # Hyperparameters
        self.L_HOMO = Config.L_HOMO
        self.K_UCG = Config.K_UCG
        self.K_ISG = Config.K_ISG
        self.MM_IMAGE_WEIGHT = Config.MM_IMAGE_WEIGHT
        self.embed_dim = Config.embed_dim
        
        # --- HETEROGENEOUS GRAPH: 2 GCN Branches (LightGCN) ---
        self.v_gcn = DragonGCN(num_users, num_items, Config.feat_dim_v, self.embed_dim, edge_index, self.device)
        self.t_gcn = DragonGCN(num_users, num_items, Config.feat_dim_t, self.embed_dim, edge_index, self.device)
        
        # --- PERSONALIZED ATTENTION WEIGHTS (FIX) ---
        # Instead of a scalar alpha, DRAGON uses a per-user weight matrix (N, 2, 1)
        # Reference: self.weight_u in hongyurain/models/dragon.py
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(self.num_users, 2, 1), gain=1.0
        ))
        
        # --- PRE-COMPUTE GRAPHS (OPTIMIZATION) ---
        print(" -> Pre-computing Homogeneous Graphs (ISG & UCG)...")
        # 1. Item Semantic Graph (ISG) - Computed once
        self.mm_adj = self._create_item_semantic_adj(feat_v, feat_t)
        
        # 2. User Co-occurrence Graph (UCG) - Converted to Sparse Matrix
        self.ucg_adj = self._create_user_graph_adj(user_graph_dict)

    def _create_item_semantic_adj(self, i_v_feat, i_t_feat):
        """Creates the fused Item Semantic Graph (ISG) Adjacency Matrix."""
        def _get_knn_adj(features, k):
            # Normalize features
            features_norm = F.normalize(features, p=2, dim=-1)
            # Cosine similarity
            sim = torch.mm(features_norm, features_norm.t())
            # Top-K
            _, knn_ind = torch.topk(sim, k, dim=-1)
            
            # Construct indices
            indices0 = torch.arange(knn_ind.shape[0], device=self.device).view(-1, 1).expand(-1, k)
            indices = torch.stack((indices0.flatten(), knn_ind.flatten()), 0)
            
            # Compute Normalized Laplacian values (simplified)
            # Here we just use binary connections for structure, or cosine sim for weights
            # The original code uses normalized laplacian of the KNN graph
            vals = torch.ones(indices.shape[1], device=self.device)
            adj = torch.sparse_coo_tensor(indices, vals, sim.size())
            
            # Row normalization (D^-1 * A)
            row_sum = torch.sparse.sum(adj, dim=1).to_dense() + 1e-7
            row_inv = torch.pow(row_sum, -1)
            norm_vals = vals * row_inv[indices[0]]
            
            return torch.sparse_coo_tensor(indices, norm_vals, sim.size())

        adj_v = _get_knn_adj(i_v_feat, self.K_ISG)
        adj_t = _get_knn_adj(i_t_feat, self.K_ISG)
        
        # Weighted Sum of Sparse Matrices (handled via dense for simplicity in setup)
        # For large scale, use coalesce on indices
        # Original code: self.mm_adj = image_adj * w + text_adj * (1-w)
        mm_adj = self.MM_IMAGE_WEIGHT * adj_v.to_dense() + (1.0 - self.MM_IMAGE_WEIGHT) * adj_t.to_dense()
        
        # Sparsify again
        indices = mm_adj.nonzero().t()
        values = mm_adj[indices[0], indices[1]]
        return torch.sparse_coo_tensor(indices, values, mm_adj.size()).to(self.device)

    def _create_user_graph_adj(self, user_graph_dict):
        """
        Converts the dictionary-based User Graph into a Sparse Adjacency Matrix
        to replace the slow loop in forward pass.
        """
        indices_list = []
        values_list = []
        
        for u_id, (neighbors, weights) in user_graph_dict.items():
            if not neighbors: continue
            
            # Softmax normalization on weights (as per original code User_Graph_sample)
            w_tensor = torch.tensor(weights, dtype=torch.float32)
            w_norm = F.softmax(w_tensor, dim=0)
            
            for neighbor, weight in zip(neighbors, w_norm):
                indices_list.append([u_id, neighbor])
                values_list.append(weight)
                
        if not indices_list:
            return torch.sparse_coo_tensor(
                torch.empty(2, 0), torch.empty(0), 
                (self.num_users, self.num_users)
            ).to(self.device)

        indices = torch.tensor(indices_list, dtype=torch.long).t()
        values = torch.tensor(values_list, dtype=torch.float32)
        
        return torch.sparse_coo_tensor(indices, values, (self.num_users, self.num_users)).to(self.device)

    def forward(self, batch_data, feat_v, feat_t):
        u_ids, pos_ids, neg_ids = batch_data
        
        # --- 1. HETEROGENEOUS GRAPH (LightGCN) ---
        u_v_het, i_v_het = self.v_gcn(feat_v)
        u_t_het, i_t_het = self.t_gcn(feat_t)
        
        # --- 2. MULTIMODAL FUSION (Attentive Concatenation) ---
        # Original Implementation: 
        # user_rep = cat(v, t) -> multiply by weight_u -> concat results
        
        # Expand dimensions for fusion (N, D) -> (N, D, 1)
        u_v_exp = u_v_het.unsqueeze(2)
        u_t_exp = u_t_het.unsqueeze(2)
        
        # Concatenate: (N, D, 2)
        u_cat = torch.cat([u_v_exp, u_t_exp], dim=2)
        
        # Attention Weighting: (N, 2, 1) -> transpose (N, 1, 2)
        # (N, D, 2) * (N, 1, 2) -> (N, D, 2) (Broadcast)
        weights = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        u_weighted = u_cat * weights
        
        # Final Concatenation: (N, D, 2) -> (N, 2D)
        u_f = torch.cat([u_weighted[:, :, 0], u_weighted[:, :, 1]], dim=1)
        
        # Item Fusion: Direct Concatenation (N, 2D)
        i_f = torch.cat([i_v_het, i_t_het], dim=1)

        # --- 3. HOMOGENEOUS GRAPHS (Dual Representation) ---
        # Propagation using Sparse Matrix Multiplication (Fast)
        h_u_homo = u_f
        h_i_homo = i_f
        
        for _ in range(self.L_HOMO):
            h_u_homo = torch.sparse.mm(self.ucg_adj, h_u_homo)
            h_i_homo = torch.sparse.mm(self.mm_adj, h_i_homo)

        # --- 4. INTEGRATION (Dual Representation) ---
        # Z = Fused (Hetero) + Homogeneous
        z_u = u_f + h_u_homo 
        z_i = i_f + h_i_homo

        # --- 5. PREDICTION AND LOSS ---
        u_emb = z_u[u_ids]
        pos_emb = z_i[pos_ids]
        neg_emb = z_i[neg_ids]
        
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # Regularization
        reg_loss = Config.weight_decay * (
            self.v_gcn.preference.pow(2).sum() + 
            self.t_gcn.preference.pow(2).sum() +
            self.weight_u.pow(2).sum()
        ) / 2.0
        
        loss = bpr_loss + reg_loss
        return loss

    def get_final_embeddings(self, feat_v, feat_t):
        with torch.no_grad():
            u_v_het, i_v_het = self.v_gcn(feat_v)
            u_t_het, i_t_het = self.t_gcn(feat_t)
            
            u_v_exp = u_v_het.unsqueeze(2)
            u_t_exp = u_t_het.unsqueeze(2)
            u_cat = torch.cat([u_v_exp, u_t_exp], dim=2)
            weights = F.softmax(self.weight_u, dim=1).transpose(1, 2)
            u_weighted = u_cat * weights
            u_f = torch.cat([u_weighted[:, :, 0], u_weighted[:, :, 1]], dim=1)
            
            i_f = torch.cat([i_v_het, i_t_het], dim=1)

            h_u_homo = u_f
            h_i_homo = i_f
            for _ in range(self.L_HOMO):
                h_u_homo = torch.sparse.mm(self.ucg_adj, h_u_homo)
                h_i_homo = torch.sparse.mm(self.mm_adj, h_i_homo)

            z_u = u_f + h_u_homo
            z_i = i_f + h_i_homo
            
            return z_u, z_i