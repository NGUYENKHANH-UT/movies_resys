from torch.utils.data import DataLoader
from .config import Config
from .dataset import MargoDataset
from .margo import MARGO
from .evaluator import Evaluator
from .trainer import Trainer
import torch
import numpy as np
import random

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
    print("MARGO TRAINING PIPELINE - FULL PIPELINE (STAGE 1 + STAGE 2)")
    print("=" * 60)
    
    # 1. Setup
    set_seed(Config.seed)
    print(f"Running on device: {Config.device}")
    print(f"Seed: {Config.seed}")
    
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
    print(f"  Training samples: {len(dataset)}")
    print(f"  Batch size: {Config.batch_size}")
    print(f"  Batches per epoch: {len(dataloader)}")
    
    # 3. Model Initialization
    print("\n[2/5] Initializing MARGO Model...")
    model = MARGO(dataset.num_users, dataset.num_items, dataset.edge_index).to(Config.device)
    print(f"  Users: {dataset.num_users}")
    print(f"  Items: {dataset.num_items}")
    print(f"  Embedding dim: {Config.embed_dim}")
    
    # 4. Evaluator & Trainer Initialization
    print("\n[3/5] Preparing evaluator and trainer...")
    evaluator = Evaluator(dataset, model)
    trainer = Trainer(model, dataloader, evaluator)
    
    # 5. Training Configuration Summary
    print("\n[4/5] Training Configuration:")
    print("-" * 60)
    print("STAGE 1 (Warm-up):")
    print(f"  Epochs: {Config.epochs_stage1}")
    print(f"  Learning Rate: {Config.lr_stage1}")
    print(f"  Modality Weights: FROZEN")
    print(f"  Alpha (Calibration): 0.0 (disabled)")
    print()
    print("STAGE 2 (Fine-tuning):")
    print(f"  Epochs: {Config.epochs_stage2}")
    print(f"  Learning Rate (GCN): {Config.lr_stage2}")
    print(f"  Learning Rate (Weights): {Config.lr_modality_weights}")
    print(f"  Modality Weights: UNFROZEN")
    print(f"  Alpha (Calibration): {Config.alpha_initial} → {Config.alpha_final}")
    print(f"  Alpha Warmup: {Config.alpha_warmup_epochs} epochs")
    print(f"  Tau (Confidence): {Config.tau}")
    print("-" * 60)
    
    # 6. Start Training Pipeline
    print("\n[5/5] Starting full training pipeline...")
    print("="*60)
    
    # This method:
    # - Trains Stage 1 with frozen weights
    # - Resets training states
    # - Trains Stage 2 with unfrozen weights and proper optimizer
    trainer.fit()
    
    # 7. Training Complete
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nCheckpoints saved:")
    print(f"  Stage 1: {Config.checkpoint_dir}/margo_best_stage1.pth")
    print(f"  Stage 2: {Config.checkpoint_dir}/margo_best_stage2.pth")
    print(f"\nBest Recall@20: {trainer.best_score:.5f}")
    print("\nNext steps:")
    print("  1. Run 'python -m src.test' to evaluate the model")
    print("  2. Run 'python -m src.inference' to test recommendations")
    print("="*60)

if __name__ == "__main__":
    main()