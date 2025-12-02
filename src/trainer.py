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
        Initialize Trainer with optimizer and training states for single-stage DRAGON training.
        """
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator
        
        # Adam with manual weight decay (only on preferences and fusion parameter)
        self.optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=0)
        
        # Unified mixed precision 
        self.scaler = GradScaler('cuda')
        self.use_amp = False

        # Early stopping parameters
        self.patience_limit = 20 # Follows DRAGON paper (stopping_step=20)
        self.best_score = -float('inf')
        self.patience_counter = 0
        
        print(f"Trainer initialized: LR={Config.lr}, Patience={self.patience_limit}")

    def save_checkpoint(self, filename, epoch=0):
        """
        Save complete checkpoint including model, optimizer, and training states.
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
        }
        
        torch.save(checkpoint, path)
        print(f"-> Checkpoint saved: {path}")

    def load_checkpoint(self, filename):
        """
        Load checkpoint for resuming training or final evaluation.
        """
        path = os.path.join(Config.checkpoint_dir, filename)
        if not os.path.exists(path):
            print(f"-> Warning: Checkpoint not found at {path}")
            return None
        
        checkpoint = torch.load(path, map_location=Config.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Restore training state only if it exists
        if 'optimizer_state_dict' in checkpoint:
             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
             self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
             self.best_score = checkpoint.get('best_score', -float('inf'))
             self.patience_counter = checkpoint.get('patience_counter', 0)
        
        print(f"-> Loaded model weights from {filename}")
        print(f"-> Resuming from epoch {checkpoint.get('epoch', 0)}")
        
        return checkpoint

    def run_training(self, num_epochs, start_epoch=0):
        """
        Train the DRAGON model.
        """
        print("\n========== START DRAGON TRAINING ==========")

        # Load features once from dataset
        feat_v = self.evaluator.dataset.feat_v
        feat_t = self.evaluator.dataset.feat_t
        
        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                u_ids, pos_ids, neg_ids = [x.to(Config.device) for x in batch]
                
                with autocast('cuda', enabled=self.use_amp):
                    loss = self.model(
                        (u_ids, pos_ids, neg_ids),
                        feat_v,
                        feat_t
                    )
                
                # Backward pass 
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
            # End of epoch statistics
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1} completed - Average loss: {avg_loss:.6f}")
            
            # Evaluation on test set (or validation set)
            metrics = self.evaluator.evaluate(k_list=[10, 20])
            recall_20 = metrics.get('Recall@20', 0.0)
            ndcg_20 = metrics.get('NDCG@20', 0.0)
            print(f"Evaluation - Recall@20: {recall_20:.5f}, NDCG@20: {ndcg_20:.5f}")
            
            # Early stopping logic
            if recall_20 > self.best_score:
                self.best_score = recall_20
                self.patience_counter = 0
                self.save_checkpoint(
                    f"{Config.model_name}.pth",
                    epoch=epoch + 1
                )
                print("New best model saved!")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{self.patience_limit}")
                if self.patience_counter >= self.patience_limit:
                    print("Early stopping triggered!")
                    break