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
    # 1. Setup
    set_seed(Config.seed)
    print(f"Running on device: {Config.device}")
    
    # 2. Data Loading
    dataset = MargoDataset()
    # Shuffle=True is crucial for training
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    # 3. Model Initialization
    print("Initializing MARGO Model...")
    model = MARGO(dataset.num_users, dataset.num_items, dataset.edge_index).to(Config.device)
    
    # 4. Evaluator & Trainer Initialization
    evaluator = Evaluator(dataset, model)
    trainer = Trainer(model, dataloader, evaluator)
    
    # 5. Start Training Pipeline
    trainer.fit()
    
    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    print("Best models are saved in ./checkpoints/")

if __name__ == "__main__":
    main()