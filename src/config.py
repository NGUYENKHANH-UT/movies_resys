import torch
import os
from dotenv import load_dotenv

# Load .env file if it exists (for Local environment)
load_dotenv()

class Config:
    # --- System ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 2024
    
    # --- Environment Detection ---
    IS_KAGGLE = os.path.exists('/kaggle/input')
    
    # --- Paths ---
    if IS_KAGGLE:
        # Kaggle Paths
        base_dir = '/kaggle/input/movies-resys-small'
        checkpoint_dir = '/kaggle/working/checkpoints'
    else:
        # Local Machine Paths
        base_dir = './ml-20m-psm'
        checkpoint_dir = './checkpoints'

    # Data Files (CSV & Pre-processed)
    ratings_path = os.path.join(base_dir, 'data/ratings.csv')
    train_path = os.path.join(base_dir, 'data/train.csv')
    valid_path = os.path.join(base_dir, 'data/valid.csv')
    test_path = os.path.join(base_dir, 'data/test.csv')
    user_graph_path = os.path.join(base_dir, 'data/user_graph_dict.npy') # NEW PATH for UCG

    # --- Milvus Config (Keep Existing) ---
    milvus_uri = os.getenv('MILVUS_URI') 
    milvus_token = os.getenv('MILVUS_TOKEN')
    milvus_collection = 'movies_multimodal'
    milvus_host = 'localhost'
    milvus_port = '19530'
    
    # --- Model Dimensions (Keep Existing) ---
    embed_dim = 64          # Latent Space Size (Model Output)
    feat_dim_v = 512        # Visual Vector Size (CLIP ViT-B/32)
    feat_dim_t = 768        # Text Vector Size (SBERT all-mpnet-base-v2)
    
    # --- Training Params (Adapted for DRAGON) ---
    batch_size = 8192
    lr = 1e-4               # Learning rate (Commonly used for DRAGON)
    weight_decay = 1e-4     # L2 regularization weight (Commonly used for DRAGON)
    epochs = 50         # Max epochs (Controlled by Early Stopping)
    
    # --- DRAGON Specifics (From DRAGON Paper) ---
    L_HETERO = 2            # Layers for Heterogeneous Graph (LightGCN)
    L_HOMO = 1              # Layers for Homogeneous Graphs (UCG & ISG)
    K_UCG = 10              # Top-K neighbors for User Co-occurrence Graph
    K_ISG = 20              # Top-K neighbors for Item Semantic Graph
    MM_IMAGE_WEIGHT = 0.1   # Alpha for weighted sum of Item Semantic Graph (Image weight)
    
    model_name = 'dragon_best'