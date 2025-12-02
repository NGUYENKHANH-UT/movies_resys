from torch.utils.data import DataLoader
from .config import Config
from .dataset import MargoDataset
from .margo import MARGO
from .evaluator import Evaluator
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
    print("MARGO MODEL EVALUATION")
    print("=" * 60)
    
    # 1. Setup
    set_seed(Config.seed)
    print(f"Running on device: {Config.device}")
    
    # 2. Data Loading
    print("\n[1/4] Loading dataset...")
    dataset = MargoDataset()
    
    # 3. Model Initialization
    print("\n[2/4] Initializing MARGO Model...")
    model = MARGO(dataset.num_users, dataset.num_items, dataset.edge_index).to(Config.device)
    
    # 4. Load Stage 2 checkpoint (final trained model)
    print("\n[3/4] Loading trained model checkpoint...")
    stage2_path = os.path.join(Config.checkpoint_dir, "margo_best_stage2.pth")
    
    # Fallback to Stage 1 if Stage 2 doesn't exist
    if not os.path.exists(stage2_path):
        print(f"Stage 2 checkpoint not found at {stage2_path}")
        stage1_path = os.path.join(Config.checkpoint_dir, "margo_best_stage1.pth")
        if os.path.exists(stage1_path):
            print(f"Loading Stage 1 checkpoint instead: {stage1_path}")
            
            checkpoint = torch.load(stage1_path, map_location=Config.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.stage = 1
        else:
            raise FileNotFoundError(
                "No trained model found! Please run train_stage1.py and train_stage2.py first."
            )
    else:
        checkpoint = torch.load(stage2_path, map_location=Config.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.stage = 2
        print(f"Successfully loaded Stage 2 model from: {stage2_path}")
    
    # Set model to evaluation mode
    model.eval()
    
    # 5. Evaluate on test set
    print("\n[4/4] Evaluating model on test set...")
    print("-" * 60)
    
    evaluator = Evaluator(dataset, model)
    
    # Evaluate with k = 5, 10, 20
    k_values = [5, 10, 20]
    print(f"\nEvaluating at K = {k_values}...\n")
    
    metrics = evaluator.evaluate(k_list=k_values)
    
    # Print results in a nice table format
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<15} {'K=5':<15} {'K=10':<15} {'K=20':<15}")
    print("-" * 60)
    
    # Recall
    recall_5 = metrics.get('Recall@5', 0.0)
    recall_10 = metrics.get('Recall@10', 0.0)
    recall_20 = metrics.get('Recall@20', 0.0)
    print(f"{'Recall':<15} {recall_5:<15.5f} {recall_10:<15.5f} {recall_20:<15.5f}")
    
    # NDCG
    ndcg_5 = metrics.get('NDCG@5', 0.0)
    ndcg_10 = metrics.get('NDCG@10', 0.0)
    ndcg_20 = metrics.get('NDCG@20', 0.0)
    print(f"{'NDCG':<15} {ndcg_5:<15.5f} {ndcg_10:<15.5f} {ndcg_20:<15.5f}")
    print("=" * 60)
    
    # Save results to file
    results_path = os.path.join(Config.checkpoint_dir, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write("MARGO Model Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model Stage: {model.stage}\n")
        f.write(f"Checkpoint: {stage2_path if os.path.exists(stage2_path) else stage1_path}\n")
        f.write("\nMetrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Recall@5:  {recall_5:.5f}\n")
        f.write(f"Recall@10: {recall_10:.5f}\n")
        f.write(f"Recall@20: {recall_20:.5f}\n")
        f.write(f"NDCG@5:    {ndcg_5:.5f}\n")
        f.write(f"NDCG@10:   {ndcg_10:.5f}\n")
        f.write(f"NDCG@20:   {ndcg_20:.5f}\n")
    
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()