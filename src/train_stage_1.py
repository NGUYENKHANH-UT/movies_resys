from torch.utils.data import DataLoader
from .config import Config
from .dataset import MargoDataset
from .margo import MARGO
from .evaluator import Evaluator
from .trainer import Trainer
import torch
import numpy as np
import random
import os

def set_seed(seed):
    """Sets random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    print("=" * 60)
    print("STAGE 1: WARM-UP TRAINING")
    print("=" * 60)
    
    # 1. Setup
    set_seed(Config.seed)
    print(f"Running on device: {Config.device}")
    
    # 2. Data Loading
    print("\n[1/5] Loading dataset...")
    dataset = MargoDataset()
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.batch_size, 
        shuffle=True,
        num_workers=2, 
        pin_memory=True
    )
    
    # 3. Model Initialization
    print("\n[2/5] Initializing MARGO Model...")
    model = MARGO(dataset.num_users, dataset.num_items, dataset.edge_index).to(Config.device)
    
    # 4. Set Stage 1 mode (freeze modality weights)
    print("\n[3/5] Configuring Stage 1 training...")
    model.stage = 1
    model.item_modality_weights.requires_grad = False
    
    # 5. Evaluator & Trainer Initialization
    print("\n[4/5] Preparing trainer...")
    evaluator = Evaluator(dataset, model)
    trainer = Trainer(model, dataloader, evaluator)
    
    # 6. Train Stage 1
    print("\n[5/5] Starting Stage 1 training...")
    trainer.run_stage("STAGE 1 (Warm-up)", Config.epochs_stage1, early_stopping=True)
    
    # 7. Save final checkpoint
    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETED")
    print("=" * 60)
    checkpoint_path = os.path.join(Config.checkpoint_dir, "margo_best_stage1.pth")
    if os.path.exists(checkpoint_path):
        print(f"Best model saved at: {checkpoint_path}")
        print("\nNext step: Run train_stage2.py to fine-tune the model")
    else:
        print("WARNING: Checkpoint not found!")

if __name__ == "__main__":
    main()