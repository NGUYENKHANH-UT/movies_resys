import torch
import os
from dotenv import load_dotenv

load_dotenv()

# --- Environment Detection (global) ---
IS_KAGGLE_ENV = os.path.exists('/kaggle/input')

try:
    import google.colab  # type: ignore
    IS_COLAB_ENV = True
except Exception:
    # fallback check env vars
    IS_COLAB_ENV = (
        'COLAB_RELEASE_TAG' in os.environ
        or 'COLAB_GPU' in os.environ
    )

class Config:
    # --- System ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 2024
    
    # --- Environment Flags ---
    IS_KAGGLE = IS_KAGGLE_ENV
    IS_COLAB = IS_COLAB_ENV
    
    # --- Paths ---
    if IS_KAGGLE:
        base_dir = '/kaggle/input/movies-resys-cleaned'
        checkpoint_dir = '/kaggle/working/checkpoints'
    elif IS_COLAB:
        # Bạn sẽ download dataset vào đây
        base_dir = '/content/movies-resys-cleaned'
        checkpoint_dir = '/content/checkpoints'
    else:
        base_dir = './ml-20m-psm'
        checkpoint_dir = './checkpoints'

    # Data Files (CSV)
    ratings_path = os.path.join(base_dir, 'data/ratings.csv')
    train_path   = os.path.join(base_dir, 'data/train.csv')
    valid_path   = os.path.join(base_dir, 'data/valid.csv')
    test_path    = os.path.join(base_dir, 'data/test.csv')
    
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
    embed_dim  = 64
    feat_dim_v = 512
    feat_dim_t = 768
    
    # --- Training Params ---
    batch_size = 4096
    
    lr_stage1 = 1e-3        # Stage 1: Higher LR for cold start
    lr_stage2 = 1e-4        # Stage 2: Lower LR for fine-tuning
    lr_modality_weights = 5e-4
    weight_decay = 1e-4
    
    epochs_stage1 = 20
    epochs_stage2 = 30
    
    # --- MARGO Specifics ---
    tau = 1.0
    alpha_initial = 0.0
    alpha_final = 0.01
    alpha_warmup_epochs = 5
    grad_clip_norm = 1.0
    
    model_name = 'margo_best'
