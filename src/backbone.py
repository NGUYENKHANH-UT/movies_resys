import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DragonGCN(nn.Module):
    """
    Simplified DRAGON Backbone: LightGCN on Heterogeneous Graph (User-Item).
    """
    def __init__(self, num_user, num_item, input_feat_dim, dim_latent, edge_index, device):
        super(DragonGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_latent = dim_latent
        self.device = device
        self.edge_index = edge_index
        self.num_nodes = num_user + num_item
        
        # 1. User Preference (Learnable Parameter)
        # Initialize with Xavier Normal for stable starting point
        self.preference = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_user, dim_latent), gain=1.0
        ))
        
        # 2. Projection MLP (Feature -> Latent)
        # Projects 512/768 vectors to 64 dimensions
        self.mlp = nn.Sequential(
            nn.Linear(input_feat_dim, 4 * dim_latent),
            nn.LeakyReLU(),
            nn.Linear(4 * dim_latent, dim_latent)
        )
        
        # Pre-calculate Normalized Adjacency Matrix
        # Used for Message Passing
        self.adj_norm = self._get_norm_adj(edge_index)

    def _get_norm_adj(self, edge_index):
        """Create sparse matrix D^-1/2 * A * D^-1/2"""
        vals = torch.ones(edge_index.size(1)).to(self.device)
        adj = torch.sparse_coo_tensor(edge_index, vals, (self.num_nodes, self.num_nodes))
        
        # Calculate Degree of each node
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
        
        # Calculate D^-0.5
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        
        # Symmetric Normalization
        src, dst = edge_index
        norm_vals = vals * d_inv_sqrt[src] * d_inv_sqrt[dst]
        
        return torch.sparse_coo_tensor(edge_index, norm_vals, (self.num_nodes, self.num_nodes))

    def forward(self, features):
        """
        features: Tensor [num_item, input_dim] retrieved from Milvus
        """
        # STEP 1: Projection
        # Transform raw features to Initial Item Embedding (E0_item)
        item_emb_0 = self.mlp(features)
        
        # STEP 2: Create Initial Node Features (E0_total)
        # Concatenate User (Preference) and Item (Projected)
        ego_embeddings = torch.cat([self.preference, item_emb_0], dim=0)
        
        # STEP 3: LightGCN Propagation
        # Propagate through 2 layers
        all_embeddings = [ego_embeddings]
        
        for k in range(2):
            # Formula: E(k+1) = Adj_norm * E(k)
            ego_embeddings = torch.sparse.mm(self.adj_norm, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        # STEP 4: Layer Combination
        # Sum or Mean of layers (DRAGON uses Sum/Mean)
        # Using Mean for stability
        final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        
        # STEP 5: Split back to User and Item
        u_final = final_embeddings[:self.num_user]
        i_final = final_embeddings[self.num_user:]
        
        return u_final, i_final