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
    
    # Fallback for Local Docker
    milvus_host = 'localhost'
    milvus_port = '19530'
    
    # --- MLflow Configuration ---
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.sean.io.vn")
    mlflow_experiment_name = "Recsys_Movies"
    
    # MLflow S3/MinIO Configuration
    mlflow_s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://apiminio.sean.io.vn/")
    mlflow_s3_ignore_tls = "true"
    
    # AWS Credentials for MinIO
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    
    # MLflow Run Configuration
    mlflow_run_name = None  # Will be set dynamically (e.g., "stage1_run1")
    mlflow_enable = False  # Global flag to enable/disable MLflow logging
    
    # --- Model Dimensions (Paper Section IV-A) ---
    embed_dim  = 64  
    feat_dim_v = 512 
    feat_dim_t = 768 
    
    # --- Training Params (Paper Section IV-A) ---
    batch_size = 16384  
    
    lr_stage1 = 1e-4  
    lr_stage2 = 1e-4 
    lr_modality_weights = 1e-4  
    
    weight_decay = 1e-4  
    
    epochs_stage1 = 20 
    epochs_stage2 = 40
    
    # --- MARGO Specifics (Paper Section IV-A) ---
    tau = 1.0  # Default value
    
    alpha_initial = 0.0
    alpha_final = 0.01 
    
    alpha_warmup_epochs = 0 
    
    grad_clip_norm = 1.0 
    
    model_name = 'margo_best'
    
    @classmethod
    def setup_mlflow_env(cls):
        """Setup environment variables for MLflow"""
        os.environ["MLFLOW_TRACKING_URI"] = cls.mlflow_tracking_uri
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = cls.mlflow_s3_endpoint
        os.environ["MLFLOW_S3_IGNORE_TLS"] = cls.mlflow_s3_ignore_tls
        os.environ["AWS_ACCESS_KEY_ID"] = cls.aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = cls.aws_secret_key