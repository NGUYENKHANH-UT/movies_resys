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
    train_path   = os.path.join(base_dir, 'data/train.csv')
    valid_path   = os.path.join(base_dir, 'data/valid.csv')
    test_path    = os.path.join(base_dir, 'data/test.csv')
    
    # --- Milvus Config ---
    milvus_uri = os.getenv('MILVUS_URI') 
    milvus_token = os.getenv('MILVUS_TOKEN')
    milvus_collection = 'movies_multimodal'
    
    # Fallback for Local Docker
    milvus_host = 'localhost'
    milvus_port = '19530'
    
    # --- Model Dimensions (Paper Section IV-A) ---
    embed_dim  = 64   # d in paper
    feat_dim_v = 512  # Visual feature dim
    feat_dim_t = 768  # Text feature dim
    
    # --- Training Params (Paper Section IV-A) ---
    batch_size = 2048  # Paper uses 2048
    
    # Paper: "We adopt the Adam optimizer and use the learning rate of 1e-4"
    # Paper does NOT mention separate LRs for different parameter groups
    lr_stage1 = 1e-4  # PAPER EXACT
    lr_stage2 = 1e-4  # PAPER EXACT
    lr_modality_weights = 1e-4  # PAPER EXACT (same as others)
    
    weight_decay = 1e-4  # β in Equation 9-10
    
    epochs_stage1 = 50  # Paper uses 100 with early stopping
    epochs_stage2 = 50
    
    # --- MARGO Specifics (Paper Section IV-A) ---
    # Paper: "We tune τ from {0.1, 1, 5, 10}"
    tau = 1.0  # Default value
    
    # Paper: "We tune α from {0, 0.01, 0.1, 1}"
    # From Figure 4 in paper: α = 0.01 works best for most datasets
    alpha_initial = 0.0
    alpha_final = 0.01  # PAPER EXACT (best value from experiments)
    
    # Paper does NOT mention warmup - apply immediately
    alpha_warmup_epochs = 0  # PAPER EXACT (no warmup)
    
    grad_clip_norm = 1.0  # Not in paper, but good practice
    
    model_name = 'margo_best'