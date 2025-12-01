import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pymilvus import connections, Collection
from tqdm import tqdm
from .config import Config

class MargoDataset(Dataset):
    def __init__(self):
        print(">>> [Dataset] Initializing...")
        
        # ==========================================
        # 1. ID MAPPING (Using ratings.csv)
        # ==========================================
        print(f" -> Reading ID space from {Config.ratings_path}...")
        # Read ratings.csv (userId, movieId, ...)
        # The original file usually has a header; add header=None if yours doesn't
        df_all = pd.read_csv(Config.ratings_path)
        
        # Get unique lists and sort to ensure deterministic mapping
        unique_users = sorted(df_all['userId'].unique())
        unique_items = sorted(df_all['movieId'].unique())
        
        self.user2id = {u: i for i, u in enumerate(unique_users)}
        self.item2id = {i: j for j, i in enumerate(unique_items)}
        
        # Reverse map for usage when querying Milvus (Internal ID -> Original Movie ID)
        self.id2item = {j: i for i, j in self.item2id.items()}
        
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        print(f" -> Stats: {self.num_users} Users, {self.num_items} Items")

        # ==========================================
        # 2. LOAD TRAIN DATA & BUILD GRAPH
        # ==========================================
        print(" -> Loading training interactions...")
        self.train_u, self.train_i = self._load_interactions(Config.train_path)
        
        print(" -> Building User-Item Graph (Edge Index)...")
        self.edge_index = self._build_graph(self.train_u, self.train_i)
        
        # ==========================================
        # 3. FETCH FEATURES FROM MILVUS
        # ==========================================
        print(" -> Fetching features from Milvus...")
        self.feat_v, self.feat_t = self._fetch_milvus_features()

    def _load_interactions(self, path):
        """Reads interaction CSV file and converts to Internal IDs"""
        print(f"   Reading {path}...")
        
        # FIX: Use header=0 to skip the header row.
        # Use names=['u', 'i', 'r', 't'] to standardize column names.
        df = pd.read_csv(path, header=0, names=['u', 'i', 'r', 't'])
        
        # Force convert columns to numeric (Integer) to avoid String type mismatch issues
        # errors='coerce' turns non-numeric values into NaN
        df['u'] = pd.to_numeric(df['u'], errors='coerce')
        df['i'] = pd.to_numeric(df['i'], errors='coerce')
        
        # Remove rows with invalid data (NaNs)
        df = df.dropna(subset=['u', 'i'])
        
        # Filter: Keep only users/items that exist in the global ID map
        original_count = len(df)
        df = df[df['u'].isin(self.user2id) & df['i'].isin(self.item2id)]
        filtered_count = len(df)
        
        if original_count != filtered_count:
            print(f"   [Warn] Filtered out {original_count - filtered_count} interactions not in ID map.")

        # Map original IDs to internal IDs
        u_ids = [self.user2id[int(u)] for u in df['u'].values]
        i_ids = [self.item2id[int(i)] for i in df['i'].values]
        
        return torch.tensor(u_ids, dtype=torch.long), torch.tensor(i_ids, dtype=torch.long)

    def _build_graph(self, u_tensor, i_tensor):
        """Create undirected User-Item graph for LightGCN"""
        # Shift Item ID: Node_Item = ID + Num_Users
        i_node_ids = i_tensor + self.num_users
        
        # Create bidirectional edges: User -> Item and Item -> User
        src = torch.cat([u_tensor, i_node_ids])
        dst = torch.cat([i_node_ids, u_tensor])
        
        return torch.stack([src, dst]).to(Config.device)

    def _fetch_milvus_features(self):
        """Query features from Milvus (Supports both Cloud and Local)"""
        try:
            if Config.milvus_uri and Config.milvus_token:
                print(f"   Connecting to Milvus Cloud: {Config.milvus_uri}")
                connections.connect("default", uri=Config.milvus_uri, token=Config.milvus_token)
            else:
                print(f"   Connecting to Local Milvus: {Config.milvus_host}:{Config.milvus_port}")
                connections.connect("default", host=Config.milvus_host, port=Config.milvus_port)
            
            col = Collection(Config.milvus_collection)
            col.load()
        except Exception as e:
            print(f"[Error] Could not connect to Milvus: {e}")
            print("Using random features for debugging...")
            return (torch.randn(self.num_items, Config.feat_dim_v).to(Config.device),
                    torch.randn(self.num_items, Config.feat_dim_t).to(Config.device))
        
        v_matrix = np.zeros((self.num_items, Config.feat_dim_v), dtype=np.float32)
        t_matrix = np.zeros((self.num_items, Config.feat_dim_t), dtype=np.float32)
        
        # Query in smaller batches
        batch_size = 100 
        all_original_ids = list(self.item2id.keys())
        milvus_data = {}
        
        print(f" -> Querying Milvus (Batch size: {batch_size})...")
        for i in tqdm(range(0, len(all_original_ids), batch_size)):
            batch_ids = all_original_ids[i : i + batch_size]
            # Convert list[int] to string for expression
            batch_ids_int = [int(x) for x in batch_ids]
            ids_str = str(batch_ids_int).replace('[', '').replace(']', '')
            
            try:
                res = col.query(
                    expr=f"movie_id in [{ids_str}]",
                    output_fields=["movie_id", "visual_emb", "text_emb"]
                )
                for item in res:
                    milvus_data[item['movie_id']] = (item['visual_emb'], item['text_emb'])
            except Exception as e:
                print(f"[Warn] Query failed for batch {i}: {e}")
                
        # Fill matrices
        missing_count = 0
        for internal_id in range(self.num_items):
            original_id = self.id2item[internal_id]
            
            if original_id in milvus_data:
                v_vec, t_vec = milvus_data[original_id]
                
                # Check for empty vectors
                if v_vec and len(v_vec) > 0:
                    v_matrix[internal_id] = v_vec
                else:
                    v_matrix[internal_id] = np.random.normal(size=Config.feat_dim_v)
                    
                if t_vec and len(t_vec) > 0:
                    t_matrix[internal_id] = t_vec
                else:
                    t_matrix[internal_id] = np.random.normal(size=Config.feat_dim_t)
            else:
                missing_count += 1
                v_matrix[internal_id] = np.random.normal(size=Config.feat_dim_v)
                t_matrix[internal_id] = np.random.normal(size=Config.feat_dim_t)
        
        print(f" -> Milvus fetch done. Missing items (Randomized): {missing_count}")
        
        return (torch.tensor(v_matrix).to(Config.device), 
                torch.tensor(t_matrix).to(Config.device))

    def __len__(self):
        return len(self.train_u)

    def __getitem__(self, idx):
        # Get User and Positive Item
        u = self.train_u[idx]
        pos_i = self.train_i[idx]
        
        # Negative Sampling (Random)
        # Randomly select an item, hoping it's negative
        # (In practice, could check more thoroughly, but for large data probability of collision is low)
        neg_i = np.random.randint(0, self.num_items)
        
        return u, pos_i, neg_i
    
    # Helper to load Valid/Test for Evaluator
    def load_eval_data(self, path):
        u, i = self._load_interactions(path)
        return u, i