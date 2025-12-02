from torch.utils.data import DataLoader
from .config import Config
from .dataset import DragonDataset
from .dragon_model import DRAGON
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
    print("DRAGON MODEL TRAINING")
    print("=" * 60)
    
    # 1. Setup
    set_seed(Config.seed)
    print(f"Running on device: {Config.device}")
    
    # 2. Data Loading
    print("\n[1/4] Loading dataset...")
    # Use DragonDataset
    dataset = DragonDataset()
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.batch_size, 
        shuffle=True,
        num_workers=2, 
        pin_memory=True
    )
    
    # 3. Model Initialization
    print("\n[2/4] Initializing DRAGON Model...")
    # Pass the pre-computed User Graph to the model
    model = DRAGON(
        dataset.num_users, 
        dataset.num_items, 
        dataset.edge_index,
        dataset.user_graph_dict
    ).to(Config.device)
    
    # 4. Evaluator & Trainer Initialization
    print("\n[3/4] Initializing Trainer and Evaluator...")
    evaluator = Evaluator(dataset, model)
    trainer = Trainer(model, dataloader, evaluator)
    
    # Check if a checkpoint exists for resuming
    checkpoint_path = os.path.join(Config.checkpoint_dir, f"{Config.model_name}.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint: {checkpoint_path}")
        checkpoint = trainer.load_checkpoint(f"{Config.model_name}.pth")
        start_epoch = checkpoint.get('epoch', 0)
    
    # 5. Start Training Pipeline
    print("\n[4/4] Starting training...")
    trainer.run_training(Config.epochs, start_epoch=start_epoch)
    
    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    print(f"Best model saved in {Config.checkpoint_dir}")

if __name__ == "__main__":
    main()