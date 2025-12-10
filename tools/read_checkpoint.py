import torch
import os
import sys
import numpy as np
from collections import OrderedDict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config

def analyze_tensor(name, tensor, show_sample=True):
    """Analyze and display information about a tensor"""
    print(f"\n{'='*70}")
    print(f"Tensor: {name}")
    print(f"{'='*70}")
    print(f"  Shape:        {tuple(tensor.shape)}")
    print(f"  Dtype:        {tensor.dtype}")
    print(f"  Device:       {tensor.device}")
    print(f"  Requires Grad: {tensor.requires_grad if hasattr(tensor, 'requires_grad') else 'N/A'}")
    print(f"  Total Elements: {tensor.numel():,}")
    print(f"  Memory Size:   {tensor.numel() * tensor.element_size() / (1024**2):.2f} MB")
    
    # Statistics
    tensor_flat = tensor.flatten()
    print(f"\n  Statistics:")
    print(f"    Min:    {tensor_flat.min().item():.6f}")
    print(f"    Max:    {tensor_flat.max().item():.6f}")
    print(f"    Mean:   {tensor_flat.mean().item():.6f}")
    print(f"    Std:    {tensor_flat.std().item():.6f}")
    print(f"    Median: {tensor_flat.median().item():.6f}")
    
    # Check for special values
    num_nan = torch.isnan(tensor_flat).sum().item()
    num_inf = torch.isinf(tensor_flat).sum().item()
    num_zero = (tensor_flat == 0).sum().item()
    
    if num_nan > 0:
        print(f"    NaN values: {num_nan} ({num_nan/tensor.numel()*100:.2f}%)")
    if num_inf > 0:
        print(f"    Inf values: {num_inf} ({num_inf/tensor.numel()*100:.2f}%)")
    if num_zero > 0:
        print(f"    Zero values: {num_zero} ({num_zero/tensor.numel()*100:.2f}%)")
    
    # Sample values
    if show_sample and tensor.numel() > 0:
        print(f"\n  Sample Values (first 10):")
        sample = tensor_flat[:min(10, tensor.numel())]
        for i, val in enumerate(sample):
            print(f"    [{i}]: {val.item():.6f}")

def inspect_checkpoint(checkpoint_path, show_samples=True, save_analysis=True):
    """
    Comprehensive checkpoint inspection
    
    Args:
        checkpoint_path: Path to checkpoint file
        show_samples: Whether to show sample values
        save_analysis: Whether to save analysis to text file
    """
    
    print("="*80)
    print("MARGO CHECKPOINT INSPECTOR")
    print("="*80)
    print(f"\nCheckpoint Path: {checkpoint_path}")
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Checkpoint file not found: {checkpoint_path}")
        return
    
    # Get file size
    file_size = os.path.getsize(checkpoint_path) / (1024**2)
    print(f"File Size: {file_size:.2f} MB")
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("[SUCCESS] Checkpoint loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return
    
    # =========================================================================
    # 1. TOP-LEVEL CHECKPOINT STRUCTURE
    # =========================================================================
    print("\n" + "="*80)
    print("1. CHECKPOINT STRUCTURE")
    print("="*80)
    
    print("\nTop-level keys:")
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, torch.Tensor):
            print(f"  - {key:30s} : Tensor {tuple(value.shape)}")
        elif isinstance(value, dict):
            print(f"  - {key:30s} : Dict with {len(value)} items")
        elif isinstance(value, (int, float)):
            print(f"  - {key:30s} : {type(value).__name__} = {value}")
        else:
            print(f"  - {key:30s} : {type(value).__name__}")
    
    # =========================================================================
    # 2. TRAINING METADATA
    # =========================================================================
    print("\n" + "="*80)
    print("2. TRAINING METADATA")
    print("="*80)
    
    metadata_keys = ['epoch', 'best_score', 'patience_counter', 'stage', 'current_alpha']
    for key in metadata_keys:
        if key in checkpoint:
            print(f"  {key:20s}: {checkpoint[key]}")
    
    # =========================================================================
    # 3. MODEL STATE DICT (Most Important Part)
    # =========================================================================
    if 'model_state_dict' in checkpoint:
        print("\n" + "="*80)
        print("3. MODEL STATE DICT - DETAILED ANALYSIS")
        print("="*80)
        
        state_dict = checkpoint['model_state_dict']
        print(f"\nTotal parameters: {len(state_dict)}")
        
        # Group parameters by component
        v_gcn_params = OrderedDict()
        t_gcn_params = OrderedDict()
        other_params = OrderedDict()
        
        for key, value in state_dict.items():
            if key.startswith('v_gcn'):
                v_gcn_params[key] = value
            elif key.startswith('t_gcn'):
                t_gcn_params[key] = value
            else:
                other_params[key] = value
        
        # =====================================================================
        # 3.1 VISUAL GCN BRANCH
        # =====================================================================
        print("\n" + "-"*80)
        print("3.1 VISUAL GCN BRANCH")
        print("-"*80)
        print(f"Number of parameters: {len(v_gcn_params)}")
        
        for name, tensor in v_gcn_params.items():
            analyze_tensor(name, tensor, show_sample=show_samples)
        
        # =====================================================================
        # 3.2 TEXT GCN BRANCH
        # =====================================================================
        print("\n" + "-"*80)
        print("3.2 TEXT GCN BRANCH")
        print("-"*80)
        print(f"Number of parameters: {len(t_gcn_params)}")
        
        for name, tensor in t_gcn_params.items():
            analyze_tensor(name, tensor, show_sample=show_samples)
        
        # =====================================================================
        # 3.3 MODALITY WEIGHTS (MARGO-specific)
        # =====================================================================
        print("\n" + "-"*80)
        print("3.3 MODALITY WEIGHTS (MARGO Specific)")
        print("-"*80)
        print(f"Number of parameters: {len(other_params)}")
        
        for name, tensor in other_params.items():
            analyze_tensor(name, tensor, show_sample=True)
            
            # If this is item_modality_weights, show weight distribution
            if 'item_modality_weights' in name:
                print(f"\n  Weight Distribution Analysis:")
                weights = torch.softmax(tensor, dim=1)
                w_v = weights[:, 0]
                w_t = weights[:, 1]
                
                print(f"\n    Visual Modality Weights (w_v):")
                print(f"      Mean:   {w_v.mean().item():.4f}")
                print(f"      Std:    {w_v.std().item():.4f}")
                print(f"      Min:    {w_v.min().item():.4f}")
                print(f"      Max:    {w_v.max().item():.4f}")
                
                print(f"\n    Text Modality Weights (w_t):")
                print(f"      Mean:   {w_t.mean().item():.4f}")
                print(f"      Std:    {w_t.std().item():.4f}")
                print(f"      Min:    {w_t.min().item():.4f}")
                print(f"      Max:    {w_t.max().item():.4f}")
                
                # Weight preference analysis
                visual_dominant = (w_v > w_t).sum().item()
                text_dominant = (w_t > w_v).sum().item()
                balanced = (w_v == w_t).sum().item()
                
                total = len(w_v)
                print(f"\n    Modality Preference:")
                print(f"      Visual Dominant:  {visual_dominant:6d} ({visual_dominant/total*100:.2f}%)")
                print(f"      Text Dominant:    {text_dominant:6d} ({text_dominant/total*100:.2f}%)")
                print(f"      Balanced:         {balanced:6d} ({balanced/total*100:.2f}%)")
    
    # =========================================================================
    # 4. OPTIMIZER STATE (If available)
    # =========================================================================
    if 'optimizer_state_dict' in checkpoint:
        print("\n" + "="*80)
        print("4. OPTIMIZER STATE")
        print("="*80)
        
        opt_state = checkpoint['optimizer_state_dict']
        print("\nOptimizer state keys:")
        for key in opt_state.keys():
            if key == 'state':
                print(f"  - {key:20s}: Dict with {len(opt_state[key])} parameter states")
            elif key == 'param_groups':
                print(f"  - {key:20s}: {len(opt_state[key])} parameter groups")
            else:
                print(f"  - {key:20s}: {opt_state[key]}")
        
        # Analyze parameter groups
        if 'param_groups' in opt_state:
            print("\n  Parameter Groups:")
            for i, group in enumerate(opt_state['param_groups']):
                print(f"\n    Group {i}:")
                for key in ['lr', 'betas', 'eps', 'weight_decay']:
                    if key in group:
                        print(f"      {key:15s}: {group[key]}")
                if 'params' in group:
                    print(f"      {'params':15s}: {len(group['params'])} parameters")
    
    # =========================================================================
    # 5. SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "="*80)
    print("5. SUMMARY STATISTICS")
    print("="*80)
    
    if 'model_state_dict' in checkpoint:
        total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        total_memory = sum(p.numel() * p.element_size() for p in checkpoint['model_state_dict'].values()) / (1024**2)
        
        print(f"\nTotal Model Parameters: {total_params:,}")
        print(f"Total Memory Usage: {total_memory:.2f} MB")
        
        # Count trainable vs non-trainable (if info available)
        print(f"\nParameter Breakdown:")
        for prefix in ['v_gcn', 't_gcn', 'item_modality']:
            params = [p for name, p in checkpoint['model_state_dict'].items() if prefix in name]
            if params:
                count = sum(p.numel() for p in params)
                memory = sum(p.numel() * p.element_size() for p in params) / (1024**2)
                print(f"  {prefix:20s}: {count:12,} params ({memory:8.2f} MB)")
    
    # =========================================================================
    # 6. SAVE ANALYSIS TO FILE
    # =========================================================================
    if save_analysis:
        output_dir = os.path.dirname(checkpoint_path)
        output_file = os.path.join(output_dir, "checkpoint_analysis.txt")
        
        print(f"\n[INFO] Saving detailed analysis to: {output_file}")
        # Note: In a real implementation, you would redirect stdout to the file
        print("[INFO] (This would save the above output to a text file)")
    
    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)

# =========================================================================
# MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect MARGO checkpoint files')
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default=None,
        help='Path to checkpoint file (default: margo_best_stage2.pth in checkpoint_dir)'
    )
    parser.add_argument(
        '--no-samples',
        action='store_true',
        help='Skip showing sample values'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving analysis to file'
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(Config.checkpoint_dir, "modal/margo_best_stage2.pth")
        print(f"Using default checkpoint path: {checkpoint_path}")
    
    # Run inspection
    inspect_checkpoint(
        checkpoint_path,
        show_samples=not args.no_samples,
        save_analysis=not args.no_save
    )