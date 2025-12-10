"""
MARGO Stage 2 Training on Modal
Compatible with Modal 1.2.4
Preserves relative imports by treating src as a package
"""
import modal
from pathlib import Path

# Setup
app = modal.App("margo-stage2-training")
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
    timeout=3600 * 5,  # 5 hours
    cpu=4,
    memory=64000,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_dotenv()],
)
def train_stage2():
    """Train Stage 2 on Modal with T4 GPU"""
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
    print("MARGO STAGE 2 TRAINING ON MODAL")
    print("=" * 60)
    print(f"Device: {Config.device}")
    print(f"Base dir: {Config.base_dir}")
    print(f"Checkpoint dir: {Config.checkpoint_dir}")
    
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
            "  python3 upload_data.py"
        )
    
    # Check checkpoints
    print("\nChecking checkpoints...")
    stage1 = '/vol/checkpoints/margo_best_stage1.pth'
    stage2 = '/vol/checkpoints/margo_best_stage2.pth'
    
    if os.path.exists(stage1):
        size = os.path.getsize(stage1)
        print(f"  ✓ Stage 1: {size / (1024*1024):.1f} MB")
    else:
        print(f"Stage 1: Not found (will train from scratch)")
    
    if os.path.exists(stage2):
        size = os.path.getsize(stage2)
        print(f"  ✓ Stage 2: {size / (1024*1024):.1f} MB (will resume)")
    else:
        print(f"Stage 2: Not found (will start fresh)")
    
    print("=" * 60)
    print()
    
    # Run as module (like python -m src.train_stage_2)
    print("Starting training as module...\n")
    
    # Change to /root so 'src' is importable
    os.chdir("/root")
    
    # Run using runpy (equivalent to python -m src.train_stage_2)
    import runpy
    runpy.run_module("src.train_stage_2", run_name="__main__")
    
    # Save volume
    print("\nCommitting volume...")
    volume.commit()
    print("Training complete and saved!")

@app.local_entrypoint()
def main():
    """Entry point: modal run modal_train.py"""
    print("=" * 60)
    print("LAUNCHING MODAL TRAINING JOB")
    print("=" * 60)
    print("GPU: T4")
    print("Timeout: 15 hours")
    print("Memory: 64 GB")
    print("=" * 60)
    print()
    
    train_stage2.remote()
    
    print()
    print("=" * 60)
    print("JOB COMPLETED")
    print("=" * 60)
    print("\nCheckpoint saved in volume: margo-model-vol")
    print("Download with: modal volume get margo-model-vol checkpoints/margo_best_stage2.pth")