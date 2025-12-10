import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import mlflow
import mlflow.pytorch
from .config import Config


class Trainer:
    def __init__(self, model, dataloader, evaluator):
        """
        Initialize Trainer with fresh optimizer and training states.
        """
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator
        
        self.optimizer = None
        self.setup_optimizer()
        
        # Mixed precision
        self.scaler = GradScaler('cuda')
        self.use_amp = False

        # Early stopping parameters
        self.patience_limit = 20
        self.best_score = -float('inf')
        self.patience_counter = 0
        
        # MLflow tracking
        self.mlflow_run = None
        self.current_stage_name = None
        
        print(f"Trainer initialized with patience={self.patience_limit}")

    def setup_optimizer(self):
        """Setup optimizer with different LRs for different stages"""
        if self.model.stage == 1:
            self.optimizer = optim.Adam(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=Config.lr_stage1,
                weight_decay=0
            )
            print(f"Stage 1 Optimizer: Adam(lr={Config.lr_stage1})")
        else:
            param_groups = [
                {
                    'params': [self.model.item_modality_weights],
                    'lr': Config.lr_modality_weights,
                    'name': 'modality_weights'
                },
                {
                    'params': list(self.model.v_gcn.parameters()) + list(self.model.t_gcn.parameters()),
                    'lr': Config.lr_stage2,
                    'name': 'gcn_params'
                }
            ]
            self.optimizer = optim.Adam(param_groups, weight_decay=0)
            print(f"Stage 2 Optimizer: Modality Weights LR={Config.lr_modality_weights}, GCN LR={Config.lr_stage2}")

    def init_mlflow(self, stage_name):
        """Initialize MLflow experiment and run"""
        if not Config.mlflow_enable:
            return
        
        try:
            # Setup environment variables
            Config.setup_mlflow_env()
            
            # Set tracking URI and experiment
            mlflow.set_tracking_uri(Config.mlflow_tracking_uri)
            mlflow.set_experiment(Config.mlflow_experiment_name)
            
            # Start new run
            run_name = f"{stage_name}_{Config.model_name}"
            self.mlflow_run = mlflow.start_run(run_name=run_name)
            self.current_stage_name = stage_name
            
            # Log hyperparameters
            mlflow.log_params({
                "model_name": Config.model_name,
                "stage": self.model.stage,
                "embed_dim": Config.embed_dim,
                "batch_size": Config.batch_size,
                "lr_stage1": Config.lr_stage1,
                "lr_stage2": Config.lr_stage2,
                "lr_modality_weights": Config.lr_modality_weights,
                "weight_decay": Config.weight_decay,
                "tau": Config.tau,
                "alpha_initial": Config.alpha_initial,
                "alpha_final": Config.alpha_final,
                "alpha_warmup_epochs": Config.alpha_warmup_epochs,
                "grad_clip_norm": Config.grad_clip_norm,
                "num_users": self.evaluator.dataset.num_users,
                "num_items": self.evaluator.dataset.num_items,
                "patience_limit": self.patience_limit,
                "seed": Config.seed
            })
            
            # Log system info
            mlflow.log_param("device", str(Config.device))
            mlflow.log_param("cuda_available", torch.cuda.is_available())
            
            print(f"✓ MLflow initialized: {Config.mlflow_tracking_uri}")
            print(f"  Experiment: {Config.mlflow_experiment_name}")
            print(f"  Run: {run_name}")
            
        except Exception as e:
            print(f"⚠ MLflow initialization failed: {e}")
            print("  Continuing without MLflow logging...")
            Config.mlflow_enable = False

    def log_metrics_to_mlflow(self, metrics_dict, step):
        """Log metrics to MLflow"""
        if not Config.mlflow_enable or self.mlflow_run is None:
            return
        
        try:
            mlflow.log_metrics(metrics_dict, step=step)
        except Exception as e:
            print(f"⚠ MLflow logging failed: {e}")

    def log_model_to_mlflow(self, checkpoint_path, stage):
        """Log PyTorch model to MLflow"""
        if not Config.mlflow_enable or self.mlflow_run is None:
            return
        
        try:
            # Log model artifact
            mlflow.log_artifact(checkpoint_path, artifact_path=f"models/stage{stage}")
            
            # Log PyTorch model (for model registry)
            mlflow.pytorch.log_model(
                self.model,
                artifact_path=f"pytorch_model_stage{stage}",
                registered_model_name=f"MARGO_Stage{stage}"
            )
            
            print(f"✓ Model logged to MLflow: stage{stage}")
        except Exception as e:
            print(f"⚠ MLflow model logging failed: {e}")

    def end_mlflow_run(self):
        """End current MLflow run"""
        if Config.mlflow_enable and self.mlflow_run is not None:
            try:
                mlflow.end_run()
                print("✓ MLflow run ended")
            except Exception as e:
                print(f"⚠ Error ending MLflow run: {e}")

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
            'current_alpha': getattr(self.model, 'current_alpha', 0.0)
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        
        # Log checkpoint to MLflow
        self.log_model_to_mlflow(path, self.model.stage)

    def load_checkpoint(self, filename, load_optimizer=True):
        """Load checkpoint with optional optimizer state"""
        path = os.path.join(Config.checkpoint_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=Config.device)
        
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
        """Alpha scheduler - gradually increase calibration loss weight"""
        if self.model.stage == 1:
            return 0.0
        
        if epoch < Config.alpha_warmup_epochs:
            return Config.alpha_initial + (Config.alpha_final - Config.alpha_initial) * (epoch / Config.alpha_warmup_epochs)
        else:
            return Config.alpha_final

    def run_stage(self, stage_name, num_epochs, early_stopping=True, start_epoch=0):
        """Train the model for one stage with improved logging"""
        print(f"\n{'='*60}")
        print(f"START {stage_name}")
        print(f"{'='*60}")
        
        # Initialize MLflow for this stage
        self.init_mlflow(stage_name)

        for epoch in range(start_epoch, num_epochs):
            if self.model.stage == 2:
                self.model.current_alpha = self.get_alpha_for_epoch(epoch)
                print(f"\nEpoch {epoch+1}: Alpha = {self.model.current_alpha:.4f}")
            
            self.model.train()
            total_loss = 0.0
            total_bpr = 0.0
            total_cal = 0.0
            total_reg = 0.0
            
            # Stage 2 specific metrics
            total_gamma_mean = 0.0
            total_gamma_min = 0.0
            total_gamma_max = 0.0
            total_gamma_zero = 0.0
            total_kl_mean = 0.0
            total_weights_v = 0.0
            total_weights_t = 0.0
            
            pbar = tqdm(self.dataloader, desc=f"{stage_name} Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                u_ids, pos_ids, neg_ids = [x.to(Config.device) for x in batch]
                
                with autocast('cuda', enabled=self.use_amp):
                    loss = self.model(
                        (u_ids, pos_ids, neg_ids),
                        self.evaluator.dataset.feat_v,
                        self.evaluator.dataset.feat_t
                    )
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Accumulate losses
                total_loss += loss.item()
                if hasattr(self.model, 'last_loss_dict'):
                    total_bpr += self.model.last_loss_dict['bpr']
                    total_cal += self.model.last_loss_dict['cal']
                    total_reg += self.model.last_loss_dict['reg']
                
                # Stage 2: Track gamma and KL metrics
                if self.model.stage == 2 and hasattr(self.model, 'last_gamma_mean'):
                    total_gamma_mean += self.model.last_gamma_mean
                    total_gamma_min += self.model.last_gamma_min
                    total_gamma_max += self.model.last_gamma_max
                    total_gamma_zero += getattr(self.model, 'last_gamma_zero', 0.0)
                    total_kl_mean += self.model.last_kl_mean
                    total_weights_v += getattr(self.model, 'last_weights_v_mean', 0.5)
                    total_weights_t += getattr(self.model, 'last_weights_t_mean', 0.5)
                
                # Update progress bar
                if self.model.stage == 2:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'bpr': f"{self.model.last_loss_dict['bpr']:.4f}",
                        'cal': f"{self.model.last_loss_dict['cal']:.4f}"
                    })
                else:
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # End of epoch statistics
            n_batches = len(self.dataloader)
            avg_loss = total_loss / n_batches
            avg_bpr = total_bpr / n_batches
            avg_cal = total_cal / n_batches
            avg_reg = total_reg / n_batches
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Total Loss: {avg_loss:.6f}")
            print(f"  BPR Loss:   {avg_bpr:.6f}")
            
            # Prepare metrics for MLflow
            mlflow_metrics = {
                "train/total_loss": avg_loss,
                "train/bpr_loss": avg_bpr,
                "train/reg_loss": avg_reg,
                "epoch": epoch + 1
            }
            
            if self.model.stage == 2:
                avg_gamma_mean = total_gamma_mean / n_batches
                avg_gamma_min = total_gamma_min / n_batches
                avg_gamma_max = total_gamma_max / n_batches
                avg_gamma_zero = total_gamma_zero / n_batches
                avg_kl = total_kl_mean / n_batches
                avg_weights_v = total_weights_v / n_batches
                avg_weights_t = total_weights_t / n_batches
                
                print(f"  Cal Loss:   {avg_cal:.6f}")
                print(f"  KL Div:     {avg_kl:.6f} (avg raw KL divergence)")
                print(f"  Gamma:      min={avg_gamma_min:.4f}, avg={avg_gamma_mean:.4f}, max={avg_gamma_max:.4f}, zeros={100*avg_gamma_zero:.1f}%")
                print(f"  Weights:    V={avg_weights_v:.4f}, T={avg_weights_t:.4f}")
                
                # Add Stage 2 specific metrics
                mlflow_metrics.update({
                    "train/cal_loss": avg_cal,
                    "train/js_divergence": avg_kl,
                    "train/gamma_mean": avg_gamma_mean,
                    "train/gamma_min": avg_gamma_min,
                    "train/gamma_max": avg_gamma_max,
                    "train/gamma_zero_pct": avg_gamma_zero * 100,
                    "train/weight_visual": avg_weights_v,
                    "train/weight_text": avg_weights_t,
                    "train/alpha": self.model.current_alpha
                })
                
            print(f"  Reg Loss:   {avg_reg:.6f}")
            print(f"{'='*60}")
            
            # Evaluation
            metrics = self.evaluator.evaluate(k_list=[5, 10, 20])
            recall_5 = metrics['Recall@5']
            recall_10 = metrics['Recall@10']
            recall_20 = metrics['Recall@20']
            ndcg_5 = metrics['NDCG@5']
            ndcg_10 = metrics['NDCG@10']
            ndcg_20 = metrics['NDCG@20']
            
            print(f"Evaluation:")
            print(f"  Recall@5:  {recall_5:.5f}  |  NDCG@5:  {ndcg_5:.5f}")
            print(f"  Recall@10: {recall_10:.5f}  |  NDCG@10: {ndcg_10:.5f}")
            print(f"  Recall@20: {recall_20:.5f}  |  NDCG@20: {ndcg_20:.5f}")
            
            # Add evaluation metrics
            mlflow_metrics.update({
                "eval/recall@5": recall_5,
                "eval/recall@10": recall_10,
                "eval/recall@20": recall_20,
                "eval/ndcg@5": ndcg_5,
                "eval/ndcg@10": ndcg_10,
                "eval/ndcg@20": ndcg_20,
                "eval/best_recall@20": self.best_score,
                "eval/patience": self.patience_counter
            })
            
            # Log all metrics to MLflow
            self.log_metrics_to_mlflow(mlflow_metrics, step=epoch + 1)
            
            # Early stopping logic
            if early_stopping:
                if recall_20 > self.best_score:
                    improvement = recall_20 - self.best_score
                    self.best_score = recall_20
                    self.patience_counter = 0
                    self.save_checkpoint(
                        f"margo_best_stage{self.model.stage}.pth",
                        epoch=epoch + 1
                    )
                    print(f"✓ New best model! Improvement: +{improvement:.5f}")
                    
                    # Log best score to MLflow
                    if Config.mlflow_enable:
                        mlflow.log_metric("best_recall@20", self.best_score)
                else:
                    self.patience_counter += 1
                    print(f"Patience: {self.patience_counter}/{self.patience_limit}")
                    if self.patience_counter >= self.patience_limit:
                        print("\n⚠ Early stopping triggered!")
                        self.load_checkpoint(
                            f"margo_best_stage{self.model.stage}.pth",
                            load_optimizer=True
                        )
                        break
        
        # End MLflow run for this stage
        self.end_mlflow_run()

    def fit(self):
        """Complete training pipeline: Stage 1 then Stage 2"""
        # Stage 1
        self.model.stage = 1
        self.model.item_modality_weights.requires_grad = False
        self.setup_optimizer()
        self.run_stage("STAGE_1_WARMUP", Config.epochs_stage1)

        print("\n" + "="*60)
        print("TRANSITIONING TO STAGE 2")
        print("="*60)
        self.best_score = -float('inf')
        self.patience_counter = 0
        print("Training states RESET")
        
        # Stage 2
        self.model.stage = 2
        self.model.item_modality_weights.requires_grad = True
        self.setup_optimizer()
        self.run_stage("STAGE_2_FINETUNE", Config.epochs_stage2)