import torch
import os
from dotenv import load_dotenv

load_dotenv()

# --- Environment Detection (global) ---
RAW_IS_KAGGLE = (
    "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    or "KAGGLE_URL_BASE" in os.environ
)

RAW_IS_COLAB = (
    'COLAB_RELEASE_TAG' in os.environ
    or 'COLAB_GPU' in os.environ
    or os.path.exists('/content/sample_data')
)

if RAW_IS_KAGGLE:
    IS_KAGGLE_ENV = True
    IS_COLAB_ENV = False
elif RAW_IS_COLAB:
    IS_KAGGLE_ENV = False
    IS_COLAB_ENV = True
else:
    IS_KAGGLE_ENV = False
    IS_COLAB_ENV = False

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
    
    lr_stage1 = 1e-3              # Stage 1: Higher LR for cold start
    lr_stage2 = 1e-4              # Stage 2: Lower LR for fine-tuning GCN
    lr_modality_weights = 5e-5    # GIẢM: LR cho weights (từ 1e-4 → 5e-5)
    weight_decay = 1e-4
    
    epochs_stage1 = 20
    epochs_stage2 = 30
    
    # --- MARGO Specifics (FINAL OPTIMIZED) ---
    tau = 2.0                     # TĂNG: 1.0 → 2.0 (less sensitive, ổn định hơn)
    alpha_initial = 0.0
    alpha_final = 0.003           # GIẢM: 0.01 → 0.003 (nhẹ nhàng hơn, tránh overwhelm BPR loss)
    alpha_warmup_epochs = 15      # TĂNG: 10 → 15 (chậm hơn để model thích nghi)
    grad_clip_norm = 1.0
    
    model_name = 'margo_best'