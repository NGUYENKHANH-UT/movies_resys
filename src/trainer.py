import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
from .config import Config


class Trainer:
    def __init__(self, model, dataloader, evaluator):
        """
        Initialize Trainer with fresh optimizer and training states.
        
        Args:
            model: MARGO model instance
            dataloader: Training data loader
            evaluator: Evaluator instance for validation
        """
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator
        
        # Adam without built-in weight decay because we apply L2 manually
        self.optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=0)
        
        # Unified mixed precision (PyTorch >= 2.0)
        self.scaler = GradScaler('cuda')
        self.use_amp = False

        # Early stopping parameters - always start with defaults
        self.patience_limit = 5
        self.best_score = -float('inf')
        self.patience_counter = 0
        
        print(f"Trainer initialized: LR={Config.lr}, Patience={self.patience_limit}")

    def save_checkpoint(self, filename, epoch=0):
        """
        Save complete checkpoint including model, optimizer, and training states.
        
        Args:
            filename: Name of checkpoint file
            epoch: Current epoch number
        """
        path = os.path.join(Config.checkpoint_dir, filename)
        os.makedirs(Config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_score': self.best_score,
            'patience_counter': self.patience_counter,
            'stage': self.model.stage
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, filename, load_optimizer=True):
        """
        Load checkpoint with optional optimizer state.
        
        Args:
            filename: Name of checkpoint file
            load_optimizer: If True, load optimizer state (for resuming training)
                          If False, only load model weights (for transfer learning)
        
        Returns:
            checkpoint dict if successful, None otherwise
        """
        path = os.path.join(Config.checkpoint_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=Config.device)
        
        # Always load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Backward compatibility with old format
            self.model.load_state_dict(checkpoint)
        
        print(f"Model weights loaded from {filename}")
        
        # Conditionally load training states
        if load_optimizer:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("  Optimizer state: RESTORED")
            
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("  Scaler state: RESTORED")
            
            self.best_score = checkpoint.get('best_score', -float('inf'))
            self.patience_counter = checkpoint.get('patience_counter', 0)
            print(f"  Best score: {self.best_score:.5f}")
            print(f"  Patience: {self.patience_counter}/{self.patience_limit}")
            
            return checkpoint
        else:
            print("  Optimizer: NOT loaded (using fresh Adam)")
            print("  Starting with default training states")
            return None

    def run_stage(self, stage_name, num_epochs, early_stopping=True, start_epoch=0):
        """
        Train the model for one stage (either warm-up or fine-tuning).
        
        Args:
            stage_name: Name of the training stage
            num_epochs: Maximum number of epochs
            early_stopping: Whether to apply early stopping
            start_epoch: Starting epoch number (for resuming)
        """
        print(f"\n========== START {stage_name} ==========")

        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f"{stage_name} Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                u_ids, pos_ids, neg_ids = [x.to(Config.device) for x in batch]
                
                # CRITICAL: Call model.forward() to compute embeddings with gradient
                # This ensures gradients flow back to GCN parameters
                with autocast('cuda', enabled=self.use_amp):
                    loss = self.model(
                        (u_ids, pos_ids, neg_ids),
                        self.evaluator.dataset.feat_v,
                        self.evaluator.dataset.feat_t
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.6f}"})
                
                # Print progress every 200 batches
                if (batch_idx + 1) % 100==0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"\nBatch {batch_idx+1}/{len(self.dataloader)} - Avg loss: {avg_loss:.6f}")
            
            # End of epoch statistics
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1} completed - Average loss: {avg_loss:.6f}")
            
            # Evaluation on test set
            metrics = self.evaluator.evaluate(k_list=[20])
            recall = metrics['Recall@20']
            ndcg = metrics['NDCG@20']
            print(f"Evaluation - Recall@20: {recall:.5f}, NDCG@20: {ndcg:.5f}")
            
            # Early stopping logic
            if early_stopping:
                if recall > self.best_score:
                    self.best_score = recall
                    self.patience_counter = 0
                    self.save_checkpoint(
                        f"margo_best_stage{self.model.stage}.pth",
                        epoch=epoch + 1
                    )
                    print("New best model saved!")
                else:
                    self.patience_counter += 1
                    print(f"Patience: {self.patience_counter}/{self.patience_limit}")
                    if self.patience_counter >= self.patience_limit:
                        print("Early stopping triggered!")
                        self.load_checkpoint(
                            f"margo_best_stage{self.model.stage}.pth",
                            load_optimizer=True
                        )
                        break

    def fit(self):
        """
        Complete training pipeline: Stage 1 (warm-up) then Stage 2 (fine-tuning).
        """
        # Stage 1: Warm-up (modality weights frozen)
        self.model.stage = 1
        self.model.item_modality_weights.requires_grad = False
        self.run_stage("STAGE 1 (Warm-up)", Config.epochs_stage1)

        # Stage 2: Reliability learning (modality weights unfrozen)
        self.model.stage = 2
        self.model.item_modality_weights.requires_grad = True
        self.run_stage("STAGE 2 (Fine-tune)", Config.epochs_stage2)