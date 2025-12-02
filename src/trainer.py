import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
from .config import Config


class Trainer:
    def __init__(self, model, dataloader, evaluator):
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator
        
        # Adam without built-in weight decay because we apply L2 manually
        self.optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=0)
        
        # Unified mixed precision (PyTorch >= 2.0)
        self.scaler = GradScaler('cuda')
        self.use_amp = True

        # Early stopping based on Recall@20
        self.patience_limit = 5
        self.best_score = -float('inf')
        self.patience_counter = 0

    def save_checkpoint(self, filename):
        path = os.path.join(Config.checkpoint_dir, filename)
        os.makedirs(Config.checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")

    def load_best_model(self, filename):
        path = os.path.join(Config.checkpoint_dir, filename)
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=Config.device))
            print("Best model loaded from checkpoint.")
        else:
            print("Warning: Checkpoint not found.")

    def run_stage(self, stage_name, num_epochs, early_stopping=True):
        """
        Train the model for one stage (either warm-up or fine-tuning).
        
        Args:
            stage_name: Name of the training stage
            num_epochs: Maximum number of epochs
            early_stopping: Whether to apply early stopping
        """
        print(f"\n========== START {stage_name} ==========")
        self.patience_counter = 0
        self.best_score = -float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f"{stage_name} Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                u_ids, pos_ids, neg_ids = [x.to(Config.device) for x in batch]
                
                # CRITICAL FIX: Call model.forward() to compute embeddings with gradient
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
                if (batch_idx + 1) % 200 == 0:
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
                    self.save_checkpoint(f"margo_best_stage{self.model.stage}.pth")
                    print("New best model saved!")
                else:
                    self.patience_counter += 1
                    print(f"Patience: {self.patience_counter}/{self.patience_limit}")
                    if self.patience_counter >= self.patience_limit:
                        print("Early stopping triggered!")
                        self.load_best_model(f"margo_best_stage{self.model.stage}.pth")
                        break

    def fit(self):
        # Stage 1: Warm-up (modality weights frozen)
        self.model.stage = 1
        self.model.item_modality_weights.requires_grad = False
        self.run_stage("STAGE 1 (Warm-up)", Config.epochs_stage1)

        # Stage 2: Reliability learning (modality weights unfrozen)
        self.model.stage = 2
        self.model.item_modality_weights.requires_grad = True
        self.run_stage("STAGE 2 (Fine-tune)", Config.epochs_stage2)