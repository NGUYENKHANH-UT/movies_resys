import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
from .config import Config


class Trainer:
    def __init__(self, model, dataloader, evaluator):
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.optimizer = optim.Adam(model.parameters(), lr=Config.lr)

        # Early stopping
        self.patience_limit = 5
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
            self.model.load_state_dict(torch.load(path, map_location=Config.device))
            print(" -> Loaded best model from checkpoint.")
        else:
            print(" -> Warning: No checkpoint found.")

    def run_stage(self, stage_name, num_epochs, early_stopping=True):
        """
        Train the model for one stage (either warm-up or fine-tuning).
        Uses efficient gradient computation: embeddings computed once per epoch WITH gradients.
        
        Args:
            stage_name: Name of the training stage for logging
            num_epochs: Maximum number of epochs to train
            early_stopping: Whether to use early stopping based on validation performance
        """
        print(f"\n========== START {stage_name} ==========")
        self.patience_counter = 0
        self.best_score = -float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            
            # COMPUTE EMBEDDINGS ONCE PER EPOCH WITH GRADIENT TRACKING
            # This approach balances speed and learning capability:
            # - Fast: GCN forward pass only once per epoch (not per batch)
            # - Learns: Maintains computational graph for gradient backpropagation
            # - Gradients from all batches accumulate into these embeddings
            u_v_all, i_v_all = self.model.v_gcn(self.evaluator.dataset.feat_v)
            u_t_all, i_t_all = self.model.t_gcn(self.evaluator.dataset.feat_t)
            
            total_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f"{stage_name} Epoch {epoch+1}/{num_epochs}")

            for batch in pbar:
                self.optimizer.zero_grad()
                u_ids, pos_ids, neg_ids = [x.to(Config.device) for x in batch]

                # LOOKUP EMBEDDINGS: This operation maintains gradient connection
                # When we call backward(), gradients will flow back through this indexing
                u_v = u_v_all[u_ids]
                u_t = u_t_all[u_ids]
                pos_v = i_v_all[pos_ids]
                pos_t = i_t_all[pos_ids]
                neg_v = i_v_all[neg_ids]
                neg_t = i_t_all[neg_ids]

                # COMPUTE COMPONENT-WISE SCORES
                # Inner product between user and item embeddings for each modality
                pos_score_v = (u_v * pos_v).sum(dim=1)
                pos_score_t = (u_t * pos_t).sum(dim=1)
                neg_score_v = (u_v * neg_v).sum(dim=1)
                neg_score_t = (u_t * neg_t).sum(dim=1)

                # SCORE FUSION based on current training stage
                if self.model.stage == 1:
                    # Stage 1 (Warm-up): Simple summation without learned weights
                    # This is equivalent to equal weighting (0.5 for each modality)
                    pos_score = pos_score_v + pos_score_t
                    neg_score = neg_score_v + neg_score_t
                else:
                    # Stage 2 (Fine-tuning): Weighted fusion with learned modality weights
                    # Softmax ensures weights sum to 1 and are non-negative
                    w_pos = F.softmax(self.model.item_modality_weights[pos_ids], dim=1)
                    w_neg = F.softmax(self.model.item_modality_weights[neg_ids], dim=1)
                    pos_score = w_pos[:, 0] * pos_score_v + w_pos[:, 1] * pos_score_t
                    neg_score = w_neg[:, 0] * neg_score_v + w_neg[:, 1] * neg_score_t

                # BPR LOSS: Bayesian Personalized Ranking
                # Objective: positive items should score higher than negative items
                # Formula: -log(sigmoid(pos_score - neg_score))
                # Numerically stable equivalent: softplus(neg_score - pos_score)
                bpr_loss = F.softplus(neg_score - pos_score).mean()

                # L2 REGULARIZATION on user preference embeddings only
                # Prevents overfitting by penalizing large embedding magnitudes
                # Note: norm(2) computes L2 norm, do NOT use pow(2) which would square it
                reg_loss = Config.weight_decay * (
                    self.model.v_gcn.preference.norm(2) +
                    self.model.t_gcn.preference.norm(2)
                )

                loss = bpr_loss + reg_loss

                # CALIBRATION LOSS: Active only in Stage 2
                # Provides explicit supervision for learning modality weights
                if self.model.stage == 2:
                    # STEP 1: Compute modality-specific ranking differences
                    # Positive diff means this modality correctly ranks pos > neg
                    diff_v = pos_score_v - neg_score_v
                    diff_t = pos_score_t - neg_score_t

                    # STEP 2: Create modality reliability vector z
                    # Apply g(x) function: keeps positive values, maps negative to -infinity
                    # Intuition: modality is reliable if it correctly ranks items
                    min_val = torch.tensor(-1e9, device=Config.device)
                    z_v = torch.where(diff_v >= 0, diff_v, min_val)
                    z_t = torch.where(diff_t >= 0, diff_t, min_val)

                    # Normalize to probability distribution (sums to 1)
                    # detach() prevents gradients from flowing into supervision signal z
                    z = F.softmax(torch.stack([z_v, z_t], dim=1), dim=1).detach()

                    # STEP 3: Compute confidence level gamma
                    # High confidence when model correctly predicts pos > neg with large margin
                    # tanh() normalizes to (-1, 1), then clamp to [0, 1] for positive confidence
                    # tau is temperature parameter controlling sensitivity
                    gamma = torch.tanh((pos_score - neg_score) / Config.tau).detach()
                    gamma = torch.clamp(gamma, min=0.0)

                    # STEP 4: Calibration via KL divergence
                    # Pushes learned weights w toward reliable modality distribution z
                    # Average weights of positive and negative items
                    w_avg = (w_pos + w_neg) / 2.0
                    
                    # KL(z || w) = sum(z * log(z/w))
                    # Measures how much z differs from w
                    # Add epsilon (1e-10) to prevent numerical issues with log(0)
                    kl_div = torch.sum(
                        z * (torch.log(z + 1e-10) - torch.log(w_avg + 1e-10)), 
                        dim=1
                    )
                    
                    # Weight KL divergence by confidence: only trust reliable predictions
                    cal_loss = torch.mean(gamma * kl_div)

                    # Add calibration loss with trade-off parameter alpha
                    loss = loss + Config.alpha * cal_loss

                # BACKWARD PASS: Compute gradients
                # Gradients flow back through:
                # 1. Loss computation
                # 2. Embedding lookups (u_v_all[u_ids], etc.)
                # 3. GCN layers (accumulated from all batches)
                loss.backward()
                
                # GRADIENT CLIPPING: Prevent exploding gradients
                # Especially important for graph neural networks with many layers
                # Clips gradient norm to max_norm if it exceeds this value
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # UPDATE PARAMETERS using computed gradients
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # EPOCH SUMMARY
            avg_loss = total_loss / len(self.dataloader)
            print(f" -> Avg Loss: {avg_loss:.4f}")

            # EVALUATION: Compute ranking metrics on test set
            metrics = self.evaluator.evaluate(k_list=[20])
            recall = metrics['Recall@20']
            ndcg = metrics['NDCG@20']
            print(f" -> Eval: Recall@20 = {recall:.5f}, NDCG@20 = {ndcg:.5f}")

            # EARLY STOPPING: Stop if validation performance plateaus
            if early_stopping:
                if recall > self.best_score:
                    # Found new best model
                    self.best_score = recall
                    self.patience_counter = 0
                    self.save_checkpoint(f"margo_best_stage{self.model.stage}.pth")
                else:
                    # No improvement
                    self.patience_counter += 1
                    print(f"    (Patience: {self.patience_counter}/{self.patience_limit})")
                    
                    if self.patience_counter >= self.patience_limit:
                        print(" -> Early Stopping Triggered!")
                        # Restore best model before exiting
                        self.load_best_model(f"margo_best_stage{self.model.stage}.pth")
                        break

    def fit(self):
        # Stage 1: Warm-up
        self.model.stage = 1
        self.model.item_modality_weights.requires_grad = False
        self.run_stage("STAGE 1 (Warm-up)", Config.epochs_stage1)

        # Stage 2: Reliability learning
        self.model.stage = 2
        self.model.item_modality_weights.requires_grad = True
        self.run_stage("STAGE 2 (Fine-tune)", Config.epochs_stage2)