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
    
    # 4. Configure Stage 2 FIRST (before creating trainer)
    print("\n[3/6] Configuring Stage 2 mode...")
    model.stage = 2
    model.item_modality_weights.requires_grad = True
    model.current_alpha = 0.0  # Will be updated by scheduler
    print("Modality weights: UNFROZEN")
    
    # 5. Create Evaluator and Trainer
    print("\n[4/6] Preparing trainer...")
    evaluator = Evaluator(dataset, model)
    trainer = Trainer(model, dataloader, evaluator)
    
    # 6. Load checkpoint
    print("\n[5/6] Loading checkpoint...")
    stage2_path = os.path.join(Config.checkpoint_dir, "margo_best_stage2.pth")
    stage1_path = os.path.join(Config.checkpoint_dir, "margo_best_stage1.pth")
    
    start_epoch = 0
    
    # CASE 1: Resume Stage 2
    if os.path.exists(stage2_path):
        print("Found existing Stage 2 checkpoint - RESUMING training...")
        checkpoint = trainer.load_checkpoint(
            "margo_best_stage2.pth", 
            load_optimizer=True
        )
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best Recall@20: {trainer.best_score:.5f}")
        print(f"Patience counter: {trainer.patience_counter}/{trainer.patience_limit}")
    
    # CASE 2: Load Stage 1 -> Start Stage 2 fresh
    elif os.path.exists(stage1_path):
        print("Loading Stage 1 checkpoint - Starting Stage 2 FRESH...")
        
        checkpoint = torch.load(stage1_path, map_location=Config.device)
        if 'model_state_dict' in checkpoint:
            # Load weights but skip optimizer/scheduler states
            state_dict = checkpoint['model_state_dict']
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
        
        print(f"Model weights loaded from: {stage1_path}")
        
        start_epoch = 0
        trainer.best_score = -float('inf')
        trainer.patience_counter = 0
        model.current_alpha = 0.0
        
        # This ensures optimizer has correct parameter groups for Stage 2
        trainer.setup_optimizer()
        
        print("\n" + "="*60)
        print("STAGE 2 INITIALIZATION COMPLETE")
        print("="*60)
        print(f"Starting from epoch: 0")
        print(f"Optimizer: FRESH (Stage 2 config)")
        print(f"Best score: RESET to -inf")
        print(f"Patience counter: RESET to 0")
        print(f"Alpha: RESET to 0.0 (will warm up)")
        print("="*60)
    
    else:
        raise FileNotFoundError(
            f"No checkpoint found!\n"
            f"Please run train_stage1.py first."
        )
    
    # 7. Debug: Print training state
    print("\n[6/6] Training state verification...")
    print("-" * 60)
    print(f"Model stage: {model.stage}")
    print(f"Start epoch: {start_epoch}")
    print(f"Best score: {trainer.best_score}")
    print(f"Patience counter: {trainer.patience_counter}")
    print(f"Current alpha: {model.current_alpha}")
    print(f"Modality weights grad: {model.item_modality_weights.requires_grad}")
    
    # Check optimizer parameter groups
    print(f"\nOptimizer parameter groups:")
    for i, param_group in enumerate(trainer.optimizer.param_groups):
        n_params = len(param_group['params'])
        lr = param_group['lr']
        name = param_group.get('name', f'group_{i}')
        print(f"  [{name}] {n_params} params, LR={lr}")
    
    # Check optimizer state
    if hasattr(trainer.optimizer, 'state') and len(trainer.optimizer.state) > 0:
        print(f"\nOptimizer has momentum/state for {len(trainer.optimizer.state)} parameters")
    else:
        print("\nOptimizer state: EMPTY (fresh start)")
    
    print("-" * 60)
    
    # 8. Train Stage 2
    print("\nStarting Stage 2 training...")
    trainer.run_stage(
        "STAGE 2 (Fine-tune)", 
        Config.epochs_stage2, 
        early_stopping=True,
        start_epoch=start_epoch
    )
    
    # 9. Final summary
    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETED")
    print("=" * 60)
    checkpoint_path = os.path.join(Config.checkpoint_dir, "margo_best_stage2.pth")
    if os.path.exists(checkpoint_path):
        print(f"Final model saved at: {checkpoint_path}")
        print(f"Best Recall@20: {trainer.best_score:.5f}")
        print("\nNext step: Run test.py to evaluate the model")
    else:
        print("WARNING: Checkpoint not found!")

if __name__ == "__main__":
    main()