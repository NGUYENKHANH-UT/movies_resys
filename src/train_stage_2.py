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
    print("STAGE 2: FINE-TUNING WITH RELIABILITY LEARNING")
    print("=" * 60)
    
    # 1. Setup
    set_seed(Config.seed)
    print(f"Running on device: {Config.device}")
    
    # 2. Data Loading
    print("\n[1/6] Loading dataset...")
    dataset = MargoDataset()
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.batch_size, 
        shuffle=True,
        num_workers=2, 
        pin_memory=True
    )
    
    # 3. Model Initialization
    print("\n[2/6] Initializing MARGO Model...")
    model = MARGO(dataset.num_users, dataset.num_items, dataset.edge_index).to(Config.device)
    
    # 4. Load checkpoint
    print("\n[3/6] Loading checkpoint...")
    stage2_path = os.path.join(Config.checkpoint_dir, "margo_best_stage2.pth")
    stage1_path = os.path.join(Config.checkpoint_dir, "margo_best_stage1.pth")
    
    # Create trainer BEFORE loading checkpoint
    evaluator = Evaluator(dataset, model)
    trainer = Trainer(model, dataloader, evaluator)
    
    start_epoch = 0  # Default starting epoch
    
    # CASE 1: Resume Stage 2 (continue training from existing checkpoint)
    if os.path.exists(stage2_path):
        print("Found existing Stage 2 checkpoint - RESUMING training...")
        checkpoint = trainer.load_checkpoint(
            "margo_best_stage2.pth", 
            load_optimizer=True  # Load optimizer state to resume properly
        )
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best Recall@20: {trainer.best_score:.5f}")
        print(f"Patience counter: {trainer.patience_counter}/{trainer.patience_limit}")
    
    # CASE 2: Load Stage 1 -> Start Stage 2 fresh
    elif os.path.exists(stage1_path):
        print("Loading Stage 1 checkpoint - Starting Stage 2 FRESH...")
        
        # Load ONLY model weights (do NOT load optimizer)
        checkpoint = torch.load(stage1_path, map_location=Config.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Backward compatibility with old checkpoint format
            model.load_state_dict(checkpoint)
        
        print(f"Model weights loaded from: {stage1_path}")
        
        # RESET all training states
        start_epoch = 0
        trainer.best_score = -float('inf')
        trainer.patience_counter = 0
        # Note: Optimizer is already fresh from Trainer.__init__()
        
        print("Starting from epoch 0")
        print("Optimizer: FRESH (new Adam instance)")
        print("Best score: RESET to -inf")
        print("Patience counter: RESET to 0")
    
    else:
        raise FileNotFoundError(
            f"No checkpoint found!\n"
            f"Please run train_stage1.py first."
        )
    
    # 5. Configure Stage 2
    print("\n[4/6] Configuring Stage 2 mode...")
    model.stage = 2
    model.item_modality_weights.requires_grad = True
    print("Modality weights: UNFROZEN")
    
    # 6. Debug: Print training state
    print("\n[5/6] Training state check...")
    print("-" * 60)
    print(f"Start epoch: {start_epoch}")
    print(f"Best score: {trainer.best_score}")
    print(f"Patience counter: {trainer.patience_counter}")
    print(f"Model stage: {model.stage}")
    print(f"Modality weights grad: {model.item_modality_weights.requires_grad}")
    
    # Check learning rate
    for i, param_group in enumerate(trainer.optimizer.param_groups):
        print(f"Param group {i} LR: {param_group['lr']}")
    
    # Check optimizer state (momentum buffers)
    if hasattr(trainer.optimizer.state, '__len__'):
        print(f"Optimizer has state for {len(trainer.optimizer.state)} parameter groups")
    else:
        print("Optimizer state: EMPTY (fresh)")
    print("-" * 60)
    
    # 7. Train Stage 2
    print("\n[6/6] Starting Stage 2 training...")
    trainer.run_stage(
        "STAGE 2 (Fine-tune)", 
        Config.epochs_stage2, 
        early_stopping=True,
        start_epoch=start_epoch
    )
    
    # 8. Save final checkpoint
    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETED")
    print("=" * 60)
    checkpoint_path = os.path.join(Config.checkpoint_dir, "margo_best_stage2.pth")
    if os.path.exists(checkpoint_path):
        print(f"Final model saved at: {checkpoint_path}")
        print("\nNext step: Run test.py to evaluate the model")
    else:
        print("WARNING: Checkpoint not found!")

if __name__ == "__main__":
    main()