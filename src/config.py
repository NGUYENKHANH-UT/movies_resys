import torch
import os
from dotenv import load_dotenv

load_dotenv()

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
        base_dir = '/kaggle/input/movies-resys-small'
        checkpoint_dir = '/kaggle/working/checkpoints'
    elif IS_COLAB:
        base_dir = '/content/movies-resys-small'
        checkpoint_dir = '/content/checkpoints'
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
    embed_dim = 64  # d in paper (default 64)
    feat_dim_v = 512  # Visual feature dimension
    feat_dim_t = 768  # Text feature dimension
    
    # --- Training Params (Paper Section IV-A) ---
    batch_size = 2048  # Paper uses 2048
    
    # Learning rates
    lr_stage1 = 1e-4  # Paper uses 1e-4 (Adam optimizer)
    lr_stage2 = 1e-4  # Same for stage 2
    lr_modality_weights = 1e-4  # Same LR for all params (paper doesn't mention separate LRs)
    
    weight_decay = 1e-4  # β in Equation 9-10 (paper default: tune from [0.01, 0.1, 1])
    
    epochs_stage1 = 50  
    epochs_stage2 = 50 
    
    # --- MARGO Specifics (Paper Section IV-A) ---
    # Paper: "We tune τ from [0.1, 1, 5, 10]"
    tau = 1.0  # Temperature for confidence (Equation 7)
    
    # Paper: "We tune α from [0, 0.01, 0.1, 1]"
    # Based on Figure 4, α = 0.01 works best for most datasets
    alpha_initial = 0.0  # Start with 0 (no calibration loss)
    alpha_final = 0.01  # Paper's best value (from experiments)
    alpha_warmup_epochs = 0  # Paper doesn't mention warmup, apply immediately
    
    # Gradient clipping (not mentioned in paper, but good practice)
    grad_clip_norm = 1.0
    
    model_name = 'margo_best'