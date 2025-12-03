import torch
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- System ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 2024
    
    # --- Environment Detection ---
    IS_KAGGLE = os.path.exists('/kaggle/input')
    
    # --- Paths ---
    if IS_KAGGLE:
        base_dir = '/kaggle/input/movies-resys-cleaned' 
        checkpoint_dir = '/kaggle/working/checkpoints'
    else:
        base_dir = './ml-20m-psm'
        checkpoint_dir = './checkpoints'

    # Data Files (CSV)
    ratings_path = os.path.join(base_dir, 'data/ratings.csv')
    train_path = os.path.join(base_dir, 'data/train.csv')
    valid_path = os.path.join(base_dir, 'data/valid.csv')
    test_path = os.path.join(base_dir, 'data/test.csv')
    
    # --- Milvus Config ---
    milvus_uri = os.getenv('MILVUS_URI') 
    milvus_token = os.getenv('MILVUS_TOKEN')
    milvus_collection = 'movies_multimodal'
    
    # --- redis / Upstash Config ---
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_password = os.getenv('REDIS_PASSWORD', None)
    redis_ssl = os.getenv('REDIS_SSL', 'False').lower() == 'true'
    
    # Fallback for Local Docker
    milvus_host = 'localhost'
    milvus_port = '19530'
    
    # --- Model Dimensions ---
    embed_dim = 64
    feat_dim_v = 512
    feat_dim_t = 768
    
    # --- Training Params ---
    batch_size = 16384
    
    lr_stage1 = 1e-3        # Stage 1: Higher LR for cold start
    lr_stage2 = 1e-4        # Stage 2: Lower LR for fine-tuning
    
    lr_modality_weights = 5e-4  # Even lower for weights
    
    weight_decay = 1e-4
    
    epochs_stage1 = 20
    epochs_stage2 = 20
    
    # --- MARGO Specifics ---
    tau = 1.0               # Increased from 0.1 → Prevent gamma saturation
    alpha_initial = 0.0     # Start with 0 (warm-up calibration loss)
    alpha_final = 0.01      # Final value (reduced from 0.1)
    alpha_warmup_epochs = 5 # Gradually increase alpha over 5 epochs
    
    grad_clip_norm = 1.0
    
    model_name = 'margo_best'