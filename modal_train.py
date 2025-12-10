"""
MARGO Full Training Pipeline (Stage 1 + Stage 2) on Modal
Compatible with Modal 1.2.4
Preserves relative imports by treating src as a package
"""
import modal
from pathlib import Path

# Setup
app = modal.App("margo-full-training")
volume = modal.Volume.from_name("margo-model-vol", create_if_missing=True)

# Build image with source code
print("Building image with source code...")
src_dir = Path(__file__).parent / "src"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
)

# Add entire src directory as a package
for py_file in src_dir.glob("*.py"):
    image = image.add_local_file(
        py_file,
        f"/root/src/{py_file.name}"
    )

# Ensure __init__.py exists
init_file = src_dir / "__init__.py"
if init_file.exists():
    image = image.add_local_file(init_file, "/root/src/__init__.py")
else:
    # Create empty __init__.py in image
    image = image.run_commands("touch /root/src/__init__.py")

@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 10,  # 10 hours for full pipeline
    cpu=4,
    memory=64000,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_dotenv()],
)
def train_full_pipeline():
    """Train Full Pipeline (Stage 1 + Stage 2) on Modal with A100 GPU"""
    import sys
    import os
    import subprocess
    
    # Add parent directory to path (so 'src' can be imported as package)
    sys.path.insert(0, "/root")
    
    # Clear environment detection
    for key in ['KAGGLE_KERNEL_RUN_TYPE', 'KAGGLE_URL_BASE', 
                'COLAB_RELEASE_TAG', 'COLAB_GPU']:
        os.environ.pop(key, None)
    
    # Import src.config to override paths
    from src.config import Config
    
    # Override paths to volume
    Config.base_dir = '/vol'
    Config.checkpoint_dir = '/vol/checkpoints'
    Config.ratings_path = '/vol/data/ratings.csv'
    Config.train_path = '/vol/data/train.csv'
    Config.valid_path = '/vol/data/valid.csv'
    Config.test_path = '/vol/data/test.csv'
    
    print("=" * 60)
    print("MARGO FULL TRAINING PIPELINE ON MODAL")
    print("Stage 1 (Warm-up) + Stage 2 (Fine-tuning)")
    print("=" * 60)
    print(f"Device: {Config.device}")
    print(f"Base dir: {Config.base_dir}")
    print(f"Checkpoint dir: {Config.checkpoint_dir}")
    print(f"MLflow Enabled: {Config.mlflow_enable}")
    
    # Verify data files
    print("\nChecking data files...")
    data_ok = True
    for name, path in [
        ("Ratings", Config.ratings_path),
        ("Train", Config.train_path),
        ("Valid", Config.valid_path),
        ("Test", Config.test_path),
    ]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_mb = size / (1024 * 1024)
            print(f"  ✓ {name}: {size_mb:.1f} MB")
        else:
            print(f"  ✗ {name}: NOT FOUND at {path}")
            data_ok = False
    
    if not data_ok:
        raise FileNotFoundError(
            "Data files missing! Upload first:\n"
            "  modal run upload_data.py"
        )
    
    # Check existing checkpoints
    print("\nChecking existing checkpoints...")
    stage1_path = '/vol/checkpoints/margo_best_stage1.pth'
    stage2_path = '/vol/checkpoints/margo_best_stage2.pth'
    
    resume_mode = False
    if os.path.exists(stage1_path):
        size = os.path.getsize(stage1_path)
        print(f"  ✓ Stage 1: {size / (1024*1024):.1f} MB (exists)")
        resume_mode = True
    else:
        print(f"  ○ Stage 1: Not found (will train from scratch)")
    
    if os.path.exists(stage2_path):
        size = os.path.getsize(stage2_path)
        print(f"  ✓ Stage 2: {size / (1024*1024):.1f} MB (exists)")
        print(f"\n  ⚠ WARNING: Stage 2 checkpoint exists!")
        print(f"  This script will train BOTH stages from scratch.")
        print(f"  If you want to resume Stage 2, use modal_train.py instead.")
        resume_mode = True
    else:
        print(f"  ○ Stage 2: Not found")
    
    if resume_mode:
        print(f"\n  Note: Existing checkpoints will be overwritten.")
        print(f"  Press Ctrl+C within 10 seconds to cancel...")
        import time
        time.sleep(10)
    
    print("=" * 60)
    print()
    
    # Training Configuration Summary
    print("Training Configuration:")
    print("-" * 60)
    print(f"Stage 1 (Warm-up):")
    print(f"  Epochs: {Config.epochs_stage1}")
    print(f"  Learning Rate: {Config.lr_stage1}")
    print(f"  Batch Size: {Config.batch_size}")
    print(f"  Modality Weights: FROZEN")
    print()
    print(f"Stage 2 (Fine-tuning):")
    print(f"  Epochs: {Config.epochs_stage2}")
    print(f"  Learning Rate (GCN): {Config.lr_stage2}")
    print(f"  Learning Rate (Weights): {Config.lr_modality_weights}")
    print(f"  Batch Size: {Config.batch_size}")
    print(f"  Modality Weights: UNFROZEN")
    print(f"  Alpha: {Config.alpha_initial} → {Config.alpha_final}")
    print(f"  Tau: {Config.tau}")
    print("-" * 60)
    print()
    
    # Run as module (like python -m src.main)
    print("Starting full training pipeline as module...\n")
    
    # Change to /root so 'src' is importable
    os.chdir("/root")
    
    # Run using runpy (equivalent to python -m src.main)
    import runpy
    runpy.run_module("src.main", run_name="__main__")
    
    # Save volume
    print("\nCommitting volume...")
    volume.commit()
    print("Training complete and saved!")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 60)
    print("\nCheckpoints saved:")
    print(f"  Stage 1: /vol/checkpoints/margo_best_stage1.pth")
    print(f"  Stage 2: /vol/checkpoints/margo_best_stage2.pth")
    print("\nDownload checkpoints:")
    print(f"  modal volume get margo-model-vol checkpoints/margo_best_stage1.pth")
    print(f"  modal volume get margo-model-vol checkpoints/margo_best_stage2.pth")
    print("=" * 60)

@app.local_entrypoint()
def main():
    """Entry point: modal run modal_train_full.py"""
    print("=" * 60)
    print("LAUNCHING MODAL FULL TRAINING JOB")
    print("=" * 60)
    print("Pipeline: Stage 1 (Warm-up) + Stage 2 (Fine-tuning)")
    print("GPU: A100")
    print("Timeout: 10 hours")
    print("Memory: 64 GB")
    print("=" * 60)
    print()
    
    # Optional: Verify Modal secrets are configured
    print("Pre-flight checks:")
    print("  ✓ Modal app initialized")
    print("  ✓ Volume configured: margo-model-vol")
    print("  ⚠ Ensure .env secrets are configured in Modal")
    print("    (MLFLOW_TRACKING_URI, MILVUS_URI, etc.)")
    print()
    print("Starting remote training job...")
    print("=" * 60)
    print()
    
    train_full_pipeline.remote()
    
    print()
    print("=" * 60)
    print("JOB COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Download checkpoints from volume")
    print("  2. View metrics on MLflow UI")
    print("  3. Run evaluation: modal run modal_test.py")
    print("=" * 60)