import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
from .config import Config


class Trainer:
    """
    Trainer following MARGO paper Algorithm 1.
    """
    def __init__(self, model, dataloader, evaluator):
        """
        Initialize Trainer with fresh optimizer and training states.
        """
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator
        
        # This will be set up in setup_optimizer()
        self.optimizer = None
        self.setup_optimizer()
        
        # Mixed precision (disabled by default, paper doesn't mention it)
        self.scaler = GradScaler('cuda')
        self.use_amp = False

        # Early stopping parameters
        self.patience_limit = 10
        self.best_score = -float('inf')
        self.patience_counter = 0
        
        print(f"Trainer initialized with patience={self.patience_limit}")

    def setup_optimizer(self):
        """
        Setup optimizer following paper:
        - Paper uses Adam with lr=1e-4 (Section IV-A)
        - Paper doesn't mention separate LRs for different parameter groups
        """
        if self.model.stage == 1:
            # Stage 1: Only optimize GCN parameters (weights are frozen)
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(
                params,
                lr=Config.lr_stage1,
                weight_decay=0  # We handle regularization manually in loss
            )
            print(f"Stage 1 Optimizer: Adam(lr={Config.lr_stage1}, params={len(params)})")
        else:
            # Stage 2: All parameters (GCN + modality weights)
            # Paper uses same LR for all parameters
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(
                params,
                lr=Config.lr_stage2,
                weight_decay=0
            )
            print(f"Stage 2 Optimizer: Adam(lr={Config.lr_stage2}, params={len(params)})")

    def save_checkpoint(self, filename, epoch=0):
        """Save complete checkpoint"""
        path = os.path.join(Config.checkpoint_dir, filename)
        os.makedirs(Config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_score': self.best_score,
            'patience_counter': self.patience_counter,
            'stage': self.model.stage,
            'current_alpha': self.model.current_alpha
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, filename, load_optimizer=True):
        """Load checkpoint with optional optimizer state"""
        path = os.path.join(Config.checkpoint_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=Config.device)
        
        # Always load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Model weights loaded from {filename}")
        
        if load_optimizer:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("  Optimizer state: RESTORED")
            
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("  Scaler state: RESTORED")
            
            self.best_score = checkpoint.get('best_score', -float('inf'))
            self.patience_counter = checkpoint.get('patience_counter', 0)
            
            if 'current_alpha' in checkpoint:
                self.model.current_alpha = checkpoint['current_alpha']
            
            print(f"  Best score: {self.best_score:.5f}")
            print(f"  Patience: {self.patience_counter}/{self.patience_limit}")
            
            return checkpoint
        else:
            print("  Optimizer: NOT loaded (using fresh Adam)")
            return None

    def get_alpha_for_epoch(self, epoch):
        """
        Alpha scheduler for calibration loss weight.
        
        Paper Section IV-A: "We tune α from [0, 0.01, 0.1, 1]"
        Paper doesn't mention warmup, but we can optionally add it.
        """
        if self.model.stage == 1:
            return 0.0
        
        if Config.alpha_warmup_epochs > 0 and epoch < Config.alpha_warmup_epochs:
            # Linear warmup from 0 to alpha_final
            ratio = epoch / Config.alpha_warmup_epochs
            return Config.alpha_initial + (Config.alpha_final - Config.alpha_initial) * ratio
        else:
            return Config.alpha_final

    def run_stage(self, stage_name, num_epochs, early_stopping=True, start_epoch=0):
        """
        Train the model for one stage following Algorithm 1 in paper.
        
        Stage 1: Optimize with L_I (Equation 9)
        Stage 2: Optimize with L_II (Equation 10)
        """
        print(f"\n{'='*60}")
        print(f"START {stage_name}")
        print(f"{'='*60}")

        for epoch in range(start_epoch, num_epochs):
            # Update alpha for Stage 2 (Equation 10)
            if self.model.stage == 2:
                self.model.current_alpha = self.get_alpha_for_epoch(epoch)
                print(f"\nEpoch {epoch+1}/{num_epochs}: Alpha = {self.model.current_alpha:.6f}")
            
            self.model.train()
            total_loss = 0.0
            total_bpr = 0.0
            total_cal = 0.0
            total_reg = 0.0
            
            pbar = tqdm(self.dataloader, desc=f"{stage_name} Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                u_ids, pos_ids, neg_ids = [x.to(Config.device) for x in batch]
                
                # Forward pass
                with autocast('cuda', enabled=self.use_amp):
                    loss = self.model(
                        (u_ids, pos_ids, neg_ids),
                        self.evaluator.dataset.feat_v,
                        self.evaluator.dataset.feat_t
                    )
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=Config.grad_clip_norm
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Accumulate losses for logging
                total_loss += loss.item()
                if hasattr(self.model, 'last_loss_dict'):
                    total_bpr += self.model.last_loss_dict['bpr']
                    total_cal += self.model.last_loss_dict['cal']
                    total_reg += self.model.last_loss_dict['reg']
                
                # Update progress bar
                if self.model.stage == 2:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'bpr': f"{self.model.last_loss_dict['bpr']:.4f}",
                        'cal': f"{self.model.last_loss_dict['cal']:.4f}"
                    })
                else:
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # ====================================================
            # End of Epoch Summary
            # ====================================================
            n_batches = len(self.dataloader)
            avg_loss = total_loss / n_batches
            avg_bpr = total_bpr / n_batches
            avg_cal = total_cal / n_batches
            avg_reg = total_reg / n_batches
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Total Loss: {avg_loss:.6f}")
            print(f"  BPR Loss:   {avg_bpr:.6f}")
            if self.model.stage == 2:
                print(f"  Cal Loss:   {avg_cal:.6f} (α={self.model.current_alpha:.4f})")
            print(f"  Reg Loss:   {avg_reg:.6f}")
            print(f"{'='*60}")
            
            # ====================================================
            # Evaluation (Paper evaluates with Recall@K and NDCG@K)
            # ====================================================
            metrics = self.evaluator.evaluate(k_list=[20])
            recall = metrics['Recall@20']
            ndcg = metrics['NDCG@20']
            print(f"Evaluation - Recall@20: {recall:.5f}, NDCG@20: {ndcg:.5f}")
            
            # ====================================================
            # Early Stopping (Paper uses early stopping strategy)
            # ====================================================
            if early_stopping:
                if recall > self.best_score:
                    improvement = recall - self.best_score
                    self.best_score = recall
                    self.patience_counter = 0
                    self.save_checkpoint(
                        f"margo_best_stage{self.model.stage}.pth",
                        epoch=epoch + 1
                    )
                    print(f"✓ New best model! Improvement: +{improvement:.5f}")
                else:
                    self.patience_counter += 1
                    print(f"✗ No improvement. Patience: {self.patience_counter}/{self.patience_limit}")
                    
                    if self.patience_counter >= self.patience_limit:
                        print("\n" + "="*60)
                        print("EARLY STOPPING TRIGGERED")
                        print("="*60)
                        print(f"Restoring best model from epoch {epoch + 1 - self.patience_limit}")
                        self.load_checkpoint(
                            f"margo_best_stage{self.model.stage}.pth",
                            load_optimizer=True
                        )
                        break

    def fit(self):
        """
        Complete training pipeline following Algorithm 1 in paper:
        
        1. Stage 1 (Lines 1-6): Pre-train without weights
        2. Stage 2 (Lines 7-13): Fine-tune with calibration
        """
        print("\n" + "="*60)
        print("MARGO TRAINING PIPELINE (Algorithm 1)")
        print("="*60)
        
        # ====================================================
        # STAGE 1: Pre-training (Algorithm 1, Lines 1-6)
        # ====================================================
        print("\n[Stage 1] Pre-training to ensure rational modality reliability...")
        self.model.stage = 1
        self.model.item_modality_weights.requires_grad = False
        self.setup_optimizer()
        self.run_stage("STAGE 1 (Pre-train)", Config.epochs_stage1, early_stopping=True)

        # ====================================================
        # Transition: Reset training states for Stage 2
        # ====================================================
        print("\n" + "="*60)
        print("TRANSITIONING TO STAGE 2")
        print("="*60)
        self.best_score = -float('inf')
        self.patience_counter = 0
        print("Training states RESET for Stage 2")
        
        # ====================================================
        # STAGE 2: Fine-tuning (Algorithm 1, Lines 7-13)
        # ====================================================
        print("\n[Stage 2] Fine-tuning with weight calibration...")
        self.model.stage = 2
        self.model.item_modality_weights.requires_grad = True
        self.setup_optimizer()
        self.run_stage("STAGE 2 (Fine-tune)", Config.epochs_stage2, early_stopping=True)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print(f"Best Recall@20: {self.best_score:.5f}")
        print("="*60)