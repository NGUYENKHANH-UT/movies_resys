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
    # Check if the code is running in a Kaggle environment
    IS_KAGGLE = os.path.exists('/kaggle/input')
    
    # --- Paths ---
    if IS_KAGGLE:
        # Kaggle Paths (Read-only Input)
        # NOTE: Replace 'my-margo-data' with your actual Kaggle dataset name
        base_dir = '/kaggle/input/movies-resys-cleaned' 
        
        # Output (Writable): Checkpoints must be saved to /kaggle/working
        checkpoint_dir = '/kaggle/working/checkpoints'
    else:
        # Local Machine Paths
        base_dir = './ml-20m-psm'
        checkpoint_dir = './checkpoints'

    # Data Files (CSV)
    ratings_path = os.path.join(base_dir, 'data/ratings.csv')
    train_path = os.path.join(base_dir, 'data/train.csv')
    valid_path = os.path.join(base_dir, 'data/valid.csv')
    test_path = os.path.join(base_dir, 'data/test.csv')
    
    # --- Milvus Config ---
    # Prioritize Environment Variables (Kaggle Secrets or .env)
    milvus_uri = os.getenv('MILVUS_URI') 
    milvus_token = os.getenv('MILVUS_TOKEN')
    milvus_collection = 'movies_multimodal'
    
    # Fallback for Local Docker (if environment variables are missing)
    milvus_host = 'localhost'
    milvus_port = '19530'
    
    # --- Model Dimensions ---
    embed_dim = 64          # Latent Space Size (Model Output)
    feat_dim_v = 512        # Visual Vector Size (CLIP ViT-B/32)
    feat_dim_t = 768        # Text Vector Size (SBERT all-mpnet-base-v2)
    
    # --- Training Params ---
    batch_size = 8192
    lr = 1e-3
    weight_decay = 1e-4
    
    # Max epochs (Controlled by Early Stopping in Trainer)
    epochs_stage1 = 50      # Warm-up Stage
    epochs_stage2 = 100      # Reliability Learning Stage
    
    # --- MARGO Specifics ---
    tau = 0.1               # Temperature (Sensitivity of Confidence function)
    alpha = 0.1             # Weight of Calibration Loss
    
    model_name = 'margo_best'