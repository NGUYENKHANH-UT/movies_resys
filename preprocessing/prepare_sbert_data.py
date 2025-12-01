import pandas as pd
import ast
import os

def prepare_sbert_data(
    input_path='./ml-20m-psm/data/movies_metadata.csv', 
    links_path='./ml-20m-psm/data/links.csv',
    output_path='./ml-20m-psm/data/movies_sbert_ready.csv'
):
    """
    Reads movies_metadata.csv, synthesizes text description.
    Maps metadata 'id' (tmdbId) to 'movieId' using links.csv.
    Saves 'movieId' and 'text' to a new CSV.
    """
    
    # 1. Check if input files exist
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return
    if not os.path.exists(links_path):
        print(f"Error: File '{links_path}' not found.")
        return

    print("Reading files...")
    try:
        # Load Metadata
        df_meta = pd.read_csv(input_path, low_memory=False)
        
        # Load Links (movieId, imdbId, tmdbId)
        df_links = pd.read_csv(links_path)
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Metadata rows: {len(df_meta)}")
    print(f"Links rows: {len(df_links)}")

    # 2. Clean IDs for Mapping
    
    # Clean Metadata 'id' (which corresponds to tmdbId)
    df_meta['clean_tmdb_id'] = pd.to_numeric(df_meta['id'], errors='coerce')
    df_meta = df_meta.dropna(subset=['clean_tmdb_id'])
    df_meta['clean_tmdb_id'] = df_meta['clean_tmdb_id'].astype(int)
    
    # Clean Links 'tmdbId'
    df_links = df_links.dropna(subset=['tmdbId'])
    df_links['tmdbId'] = df_links['tmdbId'].astype(int)

    # 3. Merge Metadata with Links to get 'movieId'
    # Inner join: Only keep movies that exist in BOTH files
    print("Mapping IDs...")
    df_merged = pd.merge(
        df_meta, 
        df_links[['movieId', 'tmdbId']], 
        left_on='clean_tmdb_id', 
        right_on='tmdbId', 
        how='inner'
    )
    
    print(f"Matched rows after merging: {len(df_merged)}")

    # 4. Helper function to parse JSON-like strings
    def extract_names(x):
        try:
            if pd.isna(x): return ""
            data = ast.literal_eval(x)
            if isinstance(data, list):
                return ", ".join([d['name'] for d in data])
            elif isinstance(data, dict):
                return data['name']
            return ""
        except:
            return ""

    print("Processing text fields...")
    
    # Fill NaNs
    df_merged['title'] = df_merged['title'].fillna('')
    df_merged['overview'] = df_merged['overview'].fillna('')
    df_merged['tagline'] = df_merged['tagline'].fillna('')
    
    if 'genres' in df_merged.columns:
        df_merged['genres_str'] = df_merged['genres'].apply(extract_names)
    else:
        df_merged['genres_str'] = ""

    # 5. Create 'text' column
    def create_text(row):
        parts = []
        if row['title']: parts.append(f"Title: {row['title']}")
        if row['genres_str']: parts.append(f"Genres: {row['genres_str']}")
        if row['tagline']: parts.append(f"Tagline: {row['tagline']}")
        if row['overview']: parts.append(f"Overview: {row['overview']}")
        return ". ".join(parts)

    df_merged['text'] = df_merged.apply(create_text, axis=1)

    # 6. Select final columns: movieId and text
    output_df = df_merged[['movieId', 'text']]
    
    # Optional: Drop empty text
    output_df = output_df[output_df['text'].str.strip() != ""]

    # 7. Save to CSV
    try:
        # Create dir if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        output_df.to_csv(output_path, index=False)
        print(f"\nSUCCESS! Saved to '{output_path}'.")
        print("Format: movieId, text")
        print(f"Final count: {len(output_df)}")
        
        print("\n--- Example Data ---")
        print(output_df.head(1))
        
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    prepare_sbert_data()