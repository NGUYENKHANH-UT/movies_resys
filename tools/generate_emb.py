import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
# Changed from CLIPVisionModel to CLIPModel to get the projected 512-dim features
from transformers import CLIPProcessor, CLIPModel 

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    # Data paths
    'text_csv_path': './ml-20m-psm/data/movies_sbert_ready.csv', 
    'posters_dir': './ml-20m-psm/posters',                                   
    'output_dir': './ml-20m-psm/data/embeddings',                 
    
    # Models
    'text_model_name': 'all-mpnet-base-v2',           
    'image_model_name': 'openai/clip-vit-base-patch32',
    
    # Processing settings
    'batch_size': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_and_validate_data(csv_path, posters_dir):
    """
    Reads the CSV file and validates the existence of corresponding poster images.
    Keeps only rows that have both valid text and an existing image file.
    """
    print("[INFO] Loading and validating data...")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)
    valid_data = []

    # Iterate through rows to check image existence
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
        movie_id = row['movieId']
        text = row['text']
        
        # Construct expected image path
        image_path = os.path.join(posters_dir, f"{movie_id}.jpg")
        
        # Condition: Image must exist AND text must not be empty
        if os.path.exists(image_path) and isinstance(text, str) and len(text) > 0:
            valid_data.append({
                'movieId': movie_id,
                'text': text,
                'image_path': image_path
            })
            
    print(f"Original rows: {len(df)}")
    print(f"Valid rows (Text + Image found): {len(valid_data)}")
    
    return pd.DataFrame(valid_data)


def generate_text_embeddings(text_list, model_name, device, batch_size):
    """
    Generates text embeddings using Sentence-BERT.
    Output dimension: 768 (for all-mpnet-base-v2)
    """
    print(f"\n[INFO] Generating Text Embeddings using {model_name}...")
    
    # Load SentenceTransformer model
    model = SentenceTransformer(model_name, device=device)
    
    # Encode texts
    # normalize_embeddings=True is important for cosine similarity / inner product search
    embeddings = model.encode(
        text_list,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True 
    )
    
    # Clean up memory
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return embeddings


def generate_image_embeddings(image_paths, model_name, device, batch_size):
    """
    Generates image embeddings using CLIP Model.
    Output dimension: 512 (Projected features)
    """
    print(f"\n[INFO] Generating Image Embeddings using {model_name}...")
    
    # Load Full CLIP Model to access the projection layer
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    all_embeddings = []
    num_samples = len(image_paths)
    
    # Process in batches to manage VRAM usage
    for i in tqdm(range(0, num_samples, batch_size), desc="Encoding Images"):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        
        # Load images in the current batch
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                # Create a black dummy image to maintain alignment if read fails
                batch_images.append(Image.new('RGB', (224, 224), color='black'))

        if not batch_images:
            continue

        # Preprocess and Move to Device
        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            # Use get_image_features to obtain the projected 512-dim embeddings
            image_features = model.get_image_features(**inputs)
            
            # Normalize embeddings (L2 norm) - Important for CLIP
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # Move to CPU and store
            all_embeddings.append(image_features.cpu().numpy())
            
    # Clean up memory
    del model
    del processor
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Concatenate all batches into a single numpy array
    return np.concatenate(all_embeddings, axis=0)


def save_results(output_dir, movie_ids, text_embs, img_embs):
    """
    Saves the generated embeddings and IDs to .npy files.
    """
    print("\n[INFO] Saving Results...")
    
    # Create directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    ids_path = os.path.join(output_dir, 'movie_ids.npy')
    text_path = os.path.join(output_dir, 'text_feat.npy')
    img_path = os.path.join(output_dir, 'image_feat.npy')
    
    # Save numpy arrays
    np.save(ids_path, movie_ids)
    np.save(text_path, text_embs)
    np.save(img_path, img_embs)
    
    print(f"Saved IDs: {ids_path} (Count: {len(movie_ids)})")
    print(f"Saved Text Embeddings: {text_path} (Shape: {text_embs.shape})")
    print(f"Saved Image Embeddings: {img_path} (Shape: {img_embs.shape})")


# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    print(f"Running on device: {CONFIG['device']}")
    
    # Step 1: Load and filter data
    df = load_and_validate_data(CONFIG['text_csv_path'], CONFIG['posters_dir'])
    
    if len(df) == 0:
        print("[ERROR] No valid data found to process. Exiting.")
        return

    # Step 2: Generate Text Embeddings
    text_embs = generate_text_embeddings(
        df['text'].tolist(), 
        CONFIG['text_model_name'], 
        CONFIG['device'], 
        CONFIG['batch_size']
    )
    
    # Step 3: Generate Image Embeddings
    img_embs = generate_image_embeddings(
        df['image_path'].tolist(), 
        CONFIG['image_model_name'], 
        CONFIG['device'], 
        CONFIG['batch_size']
    )
    
    # Step 4: Save results
    save_results(
        CONFIG['output_dir'], 
        df['movieId'].values, 
        text_embs, 
        img_embs
    )
    
    print("\n[INFO] Process completed successfully.")

if __name__ == "__main__":
    main()