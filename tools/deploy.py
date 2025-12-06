import torch
import torch.nn.functional as F
import redis
import json
import os
import sys
import numpy as np
from tqdm import tqdm
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

# Add parent directory to path to import modules from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import MargoDataset
from src.margo import MARGO

def deploy_system():
    print("============================================================")
    print("MARGO SYSTEM DEPLOYMENT (Milvus + Upstash)")
    print("============================================================")

    # ---------------------------------------------------------
    # 1. INITIALIZATION & COMPUTATION
    # ---------------------------------------------------------
    print("\n[1/3] Loading Model & Computing Final Embeddings...")
    
    # Load Data & Model
    dataset = MargoDataset()
    model = MARGO(dataset.num_users, dataset.num_items, dataset.edge_index).to(Config.device)
    
    # Load Stage 2 Checkpoint
    ckpt_path = os.path.join(Config.checkpoint_dir, "margo_best_stage2.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=Config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Compute Embeddings (LightGCN Propagation)
    with torch.no_grad():
        # Get final embeddings after GCN propagation
        (u_v, u_t), (i_v, i_t) = model.get_final_embeddings(dataset.feat_v, dataset.feat_t)
        
        # Get learned modality weights
        # Apply Softmax to ensure weights sum to 1
        weights = F.softmax(model.item_modality_weights, dim=1)
        w_v = weights[:, 0].unsqueeze(1)
        w_t = weights[:, 1].unsqueeze(1)

        # Process Items: Multiply by weights and Concatenate -> 128 dim
        # Formula: Concat(Visual * w_v, Text * w_t)
        final_items = torch.cat([i_v * w_v, i_t * w_t], dim=1).cpu().numpy()
        
        # Process Users: Concatenate only -> 128 dim
        # Formula: Concat(Visual, Text)
        final_users = torch.cat([u_v, u_t], dim=1).cpu().numpy()

    print(f"Computed: {len(final_items)} Items & {len(final_users)} Users (Dim: 128)")

    # ---------------------------------------------------------
    # 2. DEPLOY ITEMS TO MILVUS (Search Engine)
    # ---------------------------------------------------------
    print("\n[2/3] Deploying Items to Milvus...")
    
    try:
        connections.connect("default", uri=Config.milvus_uri, token=Config.milvus_token)
        
        COL_NAME = "movies_margo_final"
        
        # Drop existing collection if it exists
        if utility.has_collection(COL_NAME):
            print(f"Dropping existing collection: {COL_NAME}")
            utility.drop_collection(COL_NAME)
            
        # Define Schema: Original ID and Fused Vector (128d)
        fields = [
            FieldSchema(name="movie_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="fused_emb", dtype=DataType.FLOAT_VECTOR, dim=128)
        ]
        schema = CollectionSchema(fields, "Final Weighted Embeddings for MARGO")
        milvus_col = Collection(COL_NAME, schema)
        
        # Prepare Data: Map Internal ID -> Original MovieID
        # Ensure id2item is available in dataset
        original_item_ids = [dataset.id2item[i] for i in range(len(final_items))]
        
        # Insert Data
        print(f"Inserting {len(final_items)} items...")
        milvus_col.insert([original_item_ids, final_items])
        
        # Create Index (Metric Type: Inner Product is mandatory for weighted scoring)
        index_params = {
            "metric_type": "IP", 
            "index_type": "IVF_FLAT", 
            "params": {"nlist": 128}
        }
        milvus_col.create_index("fused_emb", index_params)
        milvus_col.load()
        
        print(f"Milvus: Successfully indexed {milvus_col.num_entities} items into '{COL_NAME}'")
        
    except Exception as e:
        print(f"Milvus Error: {e}")
        return

    # ---------------------------------------------------------
    # 3. DEPLOY USERS TO LOCAL STORAGE (Numpy Cache)
    # ---------------------------------------------------------
    print("\n[3/3] Deploying Users to Local Numpy Cache...")
    
    try:
        USER_VEC_PATH = os.path.join(Config.checkpoint_dir, "user_vectors.npy")
        np.save(USER_VEC_PATH, final_users)
        
        USER_MAP_PATH = os.path.join(Config.checkpoint_dir, "user_map.json")
        
        user_map_data = {str(k): v for k, v in dataset.user2id.items()} 

        with open(USER_MAP_PATH, 'w') as f:
            json.dump(user_map_data, f)
            
    except Exception as e:
        print(f"Local Storage Error: {e}")
        return

    print("\n============================================================")
    print("DEPLOYMENT COMPLETED SUCCESSFULLY")
    print("System is ready for inference (Using Local Numpy Cache).")
    print("============================================================")

if __name__ == "__main__":
    deploy_system()