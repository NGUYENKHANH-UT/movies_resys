import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .backbone import DragonGCN
from .config import Config

class DRAGON(nn.Module):
    """
    DRAGON Model implementation, integrating Heterogeneous and Homogeneous Graph propagation.
    """
    def __init__(self, num_users, num_items, edge_index, user_graph_dict):
        super(DRAGON, self).__init__()
        self.device = Config.device
        self.num_users = num_users
        self.num_items = num_items
        
        # Hyperparameters
        self.L_HETERO = Config.L_HETERO
        self.L_HOMO = Config.L_HOMO
        self.K_UCG = Config.K_UCG
        self.K_ISG = Config.K_ISG
        self.MM_IMAGE_WEIGHT = Config.MM_IMAGE_WEIGHT
        self.embed_dim = Config.embed_dim
        self.dim_concat = 2 * Config.embed_dim
        self.user_graph_dict = user_graph_dict

        # --- HETEROGENEOUS GRAPH: 2 GCN Branches (LightGCN) ---
        # DragonGCN implements LightGCN propagation.
        self.v_gcn = DragonGCN(num_users, num_items, Config.feat_dim_v, self.embed_dim, edge_index, self.device)
        self.t_gcn = DragonGCN(num_users, num_items, Config.feat_dim_t, self.embed_dim, edge_index, self.device)
        
        # UCG/Item Fusion Parameter (alpha for Attentive Concatenation of Users)
        # We use a single learnable alpha for UCG fusion, initialized at 0.5.
        self.user_v_alpha = nn.Parameter(torch.tensor(0.5, device=self.device))
        
    def _create_item_semantic_adj(self, i_v_feat, i_t_feat):
        """
        Creates the aggregated Item Semantic Graph (ISG) Adjacency Matrix (sparse tensor).
        """
        
        def _get_knn_adj(features, k, device):
            """Calculates KNN adjacency based on Cosine Similarity (unweighted graph)."""
            features_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
            sim = torch.matmul(features_norm, features_norm.t())
            
            # Find Top-K indices
            _, knn_ind = torch.topk(sim, k, dim=-1)
            
            # Create unweighted sparse Adj: 1 if in top-K, 0 otherwise
            adj_size = sim.size()
            indices0 = torch.arange(adj_size[0]).to(device)
            indices0 = indices0.unsqueeze(1).expand(-1, k)
            indices = torch.stack((indices0.flatten(), knn_ind.flatten()), 0)
            values = torch.ones_like(indices[0]).to(device)
            adj = torch.sparse_coo_tensor(indices, values, adj_size).coalesce()
            return adj

        adj_v = _get_knn_adj(i_v_feat, self.K_ISG, self.device)
        adj_t = _get_knn_adj(i_t_feat, self.K_ISG, self.device)
        
        # Fuse graphs by weighted summation and re-normalize (simplified D^-1 * A)
        # This implementation converts to dense temporarily to handle index overlap and summation.
        # Note: This step is memory-intensive for very large item counts.
        with torch.no_grad():
            adj_dense = self.MM_IMAGE_WEIGHT * adj_v.to_dense() + (1.0 - self.MM_IMAGE_WEIGHT) * adj_t.to_dense()
            
            # Re-sparsify
            indices = adj_dense.nonzero().t()
            values = adj_dense[indices[0], indices[1]]
            adj = torch.sparse_coo_tensor(indices, values, adj_dense.size()).coalesce()
            
            # D^-1 * A Normalization
            # The original paper uses the normalized laplacian L=D^-1 * A, so we only need D^-1
            row_sum = torch.sparse.sum(adj, dim=1).to_dense()
            d_inv = torch.pow(row_sum, -1.0)
            d_inv[torch.isinf(d_inv)] = 0.
            
            indices, values = adj.indices(), adj.values()
            norm_vals = values * d_inv[indices[0]]
            
        # Return the normalized sparse matrix for efficient message passing
        return torch.sparse_coo_tensor(indices, norm_vals, adj_dense.size()).coalesce()

    def _propagate_ucg(self, u_f):
        """
        Performs attention-based propagation on the User Co-occurrence Graph (UCG).
        This is typically a CPU-bound process due to non-standard attention aggregation.
        """
        h_u = u_f.clone()
        
        for l in range(self.L_HOMO):
            new_h_u = torch.zeros_like(h_u)
            
            # Process user-by-user (optimized for CPU/dictionary lookup)
            for u_id in range(self.num_users):
                if u_id in self.user_graph_dict and self.user_graph_dict[u_id][0]:
                    neighbors, weights = self.user_graph_dict[u_id]
                    
                    # Convert to tensors
                    neighbors = torch.tensor(neighbors, dtype=torch.long, device=self.device)
                    weights = torch.tensor(weights, dtype=torch.float, device=self.device)
                    
                    # Calculate softmax on weights
                    attn_weights = F.softmax(weights, dim=0)
                    
                    # Get neighbor embeddings h_u'^(l)
                    neighbor_h = h_u[neighbors]
                    
                    # Aggregate: sum [attn_weights * neighbor_h]
                    aggregated_h = torch.sum(attn_weights.unsqueeze(1) * neighbor_h, dim=0)
                    new_h_u[u_id] = aggregated_h

            h_u = new_h_u
            
        return h_u # h_u^(L_HOMO)
    
    def _propagate_isg(self, i_f, isg_adj):
        """
        Performs simple matrix multiplication-based propagation on the Item Semantic Graph (ISG).
        """
        h_i = i_f
        for l in range(self.L_HOMO):
            # Formula: h_i^(l+1) = Adj_norm * h_i^(l)
            h_i = torch.sparse.mm(isg_adj, h_i)
        
        return h_i # h_i^(L_HOMO)

    def forward(self, batch_data, feat_v, feat_t):
        """
        Full DRAGON forward pass to calculate the BPR loss.
        """
        u_ids, pos_ids, neg_ids = batch_data
        
        # --- 1. HETEROGENEOUS GRAPH (LightGCN) ---
        u_v_het, i_v_het = self.v_gcn(feat_v)
        u_t_het, i_t_het = self.t_gcn(feat_t)
        
        # --- 2. MULTIMODAL FUSION (Attentive Concatenation) ---
        alpha_v = torch.sigmoid(self.user_v_alpha)
        alpha_t = 1.0 - alpha_v
        
        # User Fusion: Attentive Concatenation [N_users, 2*D]
        u_f = torch.cat([alpha_v * u_v_het, alpha_t * u_t_het], dim=1) 
        
        # Item Fusion: Direct Concatenation [N_items, 2*D]
        i_f = torch.cat([i_v_het, i_t_het], dim=1)

        # --- 3. HOMOGENEOUS GRAPHS (Dual Representation) ---
        # Note: ISG Adj is re-calculated in each forward pass for consistency if features change, 
        # but in this setup, since feat_v, feat_t are frozen, caching is possible if performance is an issue.
        isg_adj = self._create_item_semantic_adj(feat_v, feat_t)
        h_i_homo = self._propagate_isg(i_f, isg_adj)
        h_u_homo = self._propagate_ucg(u_f)

        # --- 4. INTEGRATION (Dual Representation) ---
        # Final Representation Z = Fused (Hetero) + Homogeneous
        z_u = u_f + h_u_homo 
        z_i = i_f + h_i_homo

        # --- 5. PREDICTION AND LOSS ---
        u_emb, pos_emb, neg_emb = z_u[u_ids], z_i[pos_ids], z_i[neg_ids]
        
        # Scores (Inner Product)
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        
        # BPR Loss
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # Regularization (L2 on GCN Preferences and the Fusion Parameter)
        reg_loss = Config.weight_decay * (
            self.v_gcn.preference.pow(2).sum() + 
            self.t_gcn.preference.pow(2).sum() +
            self.user_v_alpha.pow(2).sum() # L2 on the learnable alpha
        ) / 2.0
        
        loss = bpr_loss + reg_loss
        
        return loss

    def get_final_embeddings(self, feat_v, feat_t):
        """
        Helper for Inference/Evaluation: Returns the final dual representations (Z_u, Z_i).
        """
        with torch.no_grad():
            # 1. HETEROGENEOUS GRAPH
            u_v_het, i_v_het = self.v_gcn(feat_v)
            u_t_het, i_t_het = self.t_gcn(feat_t)
            
            # 2. MULTIMODAL FUSION
            alpha_v = torch.sigmoid(self.user_v_alpha)
            alpha_t = 1.0 - alpha_v
            u_f = torch.cat([alpha_v * u_v_het, alpha_t * u_t_het], dim=1)
            i_f = torch.cat([i_v_het, i_t_het], dim=1)

            # 3. HOMOGENEOUS GRAPHS 
            isg_adj = self._create_item_semantic_adj(feat_v, feat_t)
            h_i_homo = self._propagate_isg(i_f, isg_adj)
            h_u_homo = self._propagate_ucg(u_f)

            # 4. INTEGRATION
            z_u = u_f + h_u_homo
            z_i = i_f + h_i_homo
            
            return z_u, z_i