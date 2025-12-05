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
    
    # --- CHANGE 1: Tăng LR cho weights để học nhanh hơn ---
    lr_modality_weights = 0 # 1e-3    # TĂNG: 1e-4 -> 1e-3 (gấp 10 lần)
    
    weight_decay = 1e-4
    
    epochs_stage1 = 30
    epochs_stage2 = 50
    
    # --- MARGO Specifics (TỐI ƯU) ---
    tau = 1.0                     # Giữ nguyên (đã tốt)
    alpha_initial = 0.0
    
    # --- CHANGE 2: Tăng Alpha để Calibration Loss có trọng lượng hơn ---
    alpha_final = 0 # 0.1             # TĂNG: 0.02 -> 0.1
    
    alpha_warmup_epochs = 8
    grad_clip_norm = 1.0
    
    model_name = 'margo_best'