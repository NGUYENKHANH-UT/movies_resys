import torch
import numpy as np
import torch.nn.functional as F
from .config import Config

class Evaluator:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.device = Config.device
        
        # 1. Prepare Masking Data
        # We need to mask items that the user has already interacted with in Train set
        # Dict: {user_id: [item_id1, item_id2...]} (Internal IDs)
        print("Preparing Evaluator data...")
        self.user_pos_train = self._build_user_pos_dict(dataset.train_u, dataset.train_i)
        
        # 2. Load Test Data (Ground Truth)
        # Using the helper function from dataset
        test_u, test_i = dataset.load_eval_data(Config.test_path)
        self.test_user_pos = self._build_user_pos_dict(test_u, test_i)
        self.test_users = list(self.test_user_pos.keys())
        print(f"Evaluator ready. Test users: {len(self.test_users)}")

    def _build_user_pos_dict(self, u_tensor, i_tensor):
        user_pos = {}
        u_list = u_tensor.tolist()
        i_list = i_tensor.tolist()
        for u, i in zip(u_list, i_list):
            if u not in user_pos: user_pos[u] = []
            user_pos[u].append(i)
        return user_pos

    def evaluate(self, k_list=[20]):
        self.model.eval()
        
        # 1. Get All Final Embeddings (User Z_u & Item Z_i)
        # DRAGON returns the single, fused, and propagated dual representations (Z_u, Z_i).
        # FIX: Unpack directly into two tensors (z_u, z_i) instead of the nested tuple format used by MARGO.
        z_u, z_i = self.model.get_final_embeddings(
            self.dataset.feat_v, self.dataset.feat_t
        )
        
        results = {f'Recall@{k}': 0.0 for k in k_list}
        results.update({f'NDCG@{k}': 0.0 for k in k_list})
        
        # 2. Batch Evaluation (to avoid OOM)
        batch_size = 128
        num_test_users = len(self.test_users)
        
        with torch.no_grad():
            for i in range(0, num_test_users, batch_size):
                batch_users = self.test_users[i : i + batch_size]
                batch_u_tensor = torch.tensor(batch_users).to(self.device)
                
                # --- A. Calculate Score Matrix (Full Ranking) ---
                # Get user vectors for batch (Z_u)
                batch_z_u = z_u[batch_u_tensor]
                
                # Matrix Mult: User Z_u x Item Z_i^T
                # [Batch, 2*Dim] x [2*Dim, N_items] -> [Batch, N_items]
                batch_scores = torch.matmul(batch_z_u, z_i.t())
                
                # --- B. Masking (Hide trained items) ---
                # Assign -inf to items present in training set
                for idx, u_id in enumerate(batch_users):
                    if u_id in self.user_pos_train:
                        pos_items = self.user_pos_train[u_id]
                        batch_scores[idx][pos_items] = -float('inf')
                
                # --- C. Metrics Calculation ---
                # Get Top-K largest scores
                max_k = max(k_list)
                _, topk_indices = torch.topk(batch_scores, max_k)
                topk_indices = topk_indices.cpu().numpy()
                
                for idx, u_id in enumerate(batch_users):
                    ground_truth = self.test_user_pos[u_id]
                    pred_items = topk_indices[idx]
                    
                    for k in k_list:
                        pred_k = pred_items[:k]
                        # Hits: intersection of prediction and ground truth
                        hits = len(set(pred_k) & set(ground_truth))
                        
                        # Recall
                        results[f'Recall@{k}'] += hits / len(ground_truth)
                        
                        # NDCG
                        dcg = 0.0
                        idcg = 0.0
                        
                        # Calculate DCG
                        for j, item in enumerate(pred_k):
                            if item in ground_truth:
                                dcg += 1.0 / np.log2(j + 2)
                        
                        # Calculate IDCG (Ideal DCG)
                        for j in range(min(len(ground_truth), k)):
                            idcg += 1.0 / np.log2(j + 2)
                            
                        results[f'NDCG@{k}'] += dcg / idcg if idcg > 0 else 0
        
        # Average metrics
        for k in results:
            results[k] /= num_test_users
            
        return results