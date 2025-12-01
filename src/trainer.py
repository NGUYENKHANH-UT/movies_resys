import torch
import torch.optim as optim
from tqdm import tqdm
import os
from .config import Config

class Trainer:
    def __init__(self, model, dataloader, evaluator):
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.optimizer = optim.Adam(model.parameters(), lr=Config.lr)
        
        # --- Early Stopping State ---
        self.patience_limit = 5 # Stop if no improvement for 5 evaluations
        self.best_score = -float('inf')
        self.patience_counter = 0

    def save_checkpoint(self, filename):
        path = os.path.join(Config.checkpoint_dir, filename)
        os.makedirs(Config.checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f" -> Checkpoint saved: {path}")

    def load_best_model(self, filename):
        path = os.path.join(Config.checkpoint_dir, filename)
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print(" -> Loaded best model from checkpoint.")
        else:
            print(" -> Warning: No checkpoint found to load.")

    def run_stage(self, stage_name, num_epochs, early_stopping=True):
        """Runs a specific training stage (1 or 2)"""
        print(f"\n========== START {stage_name} ==========")
        
        # Reset patience for new stage
        self.patience_counter = 0
        self.best_score = -float('inf')
        
        for epoch in range(num_epochs):
            # --- 1. TRAINING LOOP ---
            self.model.train()
            total_loss = 0
            
            pbar = tqdm(self.dataloader, desc=f"{stage_name} Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                # Move batch data to GPU
                batch = [x.to(Config.device) for x in batch]
                
                self.optimizer.zero_grad()
                
                # Forward Pass (Calculate Loss)
                loss = self.model(batch, self.evaluator.dataset.feat_v, self.evaluator.dataset.feat_t)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(self.dataloader)
            print(f" -> Avg Loss: {avg_loss:.4f}")
            
            # --- 2. EVALUATION & EARLY STOPPING ---
            # Evaluate every epoch
            metrics = self.evaluator.evaluate(k_list=[20])
            recall = metrics['Recall@20']
            ndcg = metrics['NDCG@20']
            
            print(f" -> Eval: Recall@20 = {recall:.4f}, NDCG@20 = {ndcg:.4f}")
            
            if early_stopping:
                # If better than previous best
                if recall > self.best_score:
                    self.best_score = recall
                    self.patience_counter = 0
                    # Save best model for this stage
                    self.save_checkpoint(f"{Config.model_name}_stage{self.model.stage}.pth")
                else:
                    # If not better
                    self.patience_counter += 1
                    print(f"    (Patience: {self.patience_counter}/{self.patience_limit})")
                    
                    # Trigger Stop
                    if self.patience_counter >= self.patience_limit:
                        print(" -> Early Stopping Triggered! Moving to next step.")
                        # Restore best model
                        self.load_best_model(f"{Config.model_name}_stage{self.model.stage}.pth")
                        break

    def fit(self):
        # --- STAGE 1: WARM-UP ---
        # Goal: Train Backbone, Freeze MARGO weights
        self.model.stage = 1
        self.model.item_modality_weights.requires_grad = False 
        
        self.run_stage("STAGE 1 (Warm-up)", Config.epochs_stage1, early_stopping=True)
        
        # --- STAGE 2: RELIABILITY LEARNING ---
        # Goal: Learn weights W + Fine-tune backbone
        self.model.stage = 2
        self.model.item_modality_weights.requires_grad = True # Unfreeze
        
        self.run_stage("STAGE 2 (Fine-tune)", Config.epochs_stage2, early_stopping=True)