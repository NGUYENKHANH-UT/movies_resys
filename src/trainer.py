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
        print(f"\n========== START {stage_name} ==========")
        self.patience_counter = 0
        self.best_score = -float('inf')

        for epoch in range(num_epochs):
            self.model.train()

            # Pre-compute all embeddings once per epoch (decoupled training)
            with torch.no_grad():
                u_v_all, i_v_all = self.model.v_gcn(self.evaluator.dataset.feat_v)
                u_t_all, i_t_all = self.model.t_gcn(self.evaluator.dataset.feat_t)

            total_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f"{stage_name} Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                u_ids, pos_ids, neg_ids = [x.to(Config.device) for x in batch]

                with autocast('cuda', enabled=self.use_amp):
                    # Lookup pre-computed embeddings
                    u_v = u_v_all[u_ids]
                    u_t = u_t_all[u_ids]
                    pos_v = i_v_all[pos_ids]
                    pos_t = i_t_all[pos_ids]
                    neg_v = i_v_all[neg_ids]
                    neg_t = i_t_all[neg_ids]

                    # Component scores
                    pos_score_v = (u_v * pos_v).sum(dim=1)
                    pos_score_t = (u_t * pos_t).sum(dim=1)
                    neg_score_v = (u_v * neg_v).sum(dim=1)
                    neg_score_t = (u_t * neg_t).sum(dim=1)

                    # Score fusion
                    if self.model.stage == 1:
                        pos_score = pos_score_v + pos_score_t
                        neg_score = neg_score_v + neg_score_t
                    else:
                        w_pos = F.softmax(self.model.item_modality_weights[pos_ids], dim=1)
                        w_neg = F.softmax(self.model.item_modality_weights[neg_ids], dim=1)
                        pos_score = w_pos[:, 0] * pos_score_v + w_pos[:, 1] * pos_score_t
                        neg_score = w_neg[:, 0] * neg_score_v + w_neg[:, 1] * neg_score_t

                    # BPR loss
                    bpr_loss = F.softplus(neg_score - pos_score).mean()

                    # L2 regularization on user preference parameters only
                    reg_loss = Config.weight_decay * (
                        self.model.v_gcn.preference.pow(2).sum() +
                        self.model.t_gcn.preference.pow(2).sum()
                    ) / 2.0

                    loss = bpr_loss + reg_loss

                    # Calibration loss (Stage 2 only)
                    if self.model.stage == 2:
                        diff_v = pos_score_v - neg_score_v
                        diff_t = pos_score_t - neg_score_t

                        min_val = torch.tensor(-1e9, device=Config.device)
                        z_v = torch.where(diff_v >= 0, diff_v, min_val)
                        z_t = torch.where(diff_t >= 0, diff_t, min_val)

                        z = F.softmax(torch.stack([z_v, z_t], dim=1), dim=1).detach()

                        gamma = torch.tanh((pos_score - neg_score) / Config.tau).detach()
                        gamma = torch.clamp(gamma, min=0.0)

                        w_avg = (w_pos + w_neg) / 2.0
                        kl_div = torch.sum(z * (torch.log(z + 1e-10) - torch.log(w_avg + 1e-10)), dim=1)
                        cal_loss = torch.mean(gamma * kl_div)

                        loss = loss + Config.alpha * cal_loss

                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.6f}"})

                # Print average loss every 200 batches for monitoring
                if (batch_idx + 1) % 200 == 0:
                    avg_so_far = total_loss / (batch_idx + 1)
                    print(f"Batch {batch_idx+1}/{len(self.dataloader)} - Avg loss so far: {avg_so_far:.6f}")

            # End of epoch statistics
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1} completed - Average loss: {avg_loss:.6f}")

            # Full evaluation on test set (once per epoch)
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