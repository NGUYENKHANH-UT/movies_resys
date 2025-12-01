import pandas as pd
import os

def check_tmdb_in_metadata(links_path='links.csv', metadata_path='movies_metadata.csv'):
    """
    Checks if tmdbIds from the links file exist in the metadata file (in the 'id' column).
    """
    
    # 1. Check if files exist
    if not os.path.exists(links_path):
        print(f"Error: File '{links_path}' not found")
        return
    if not os.path.exists(metadata_path):
        print(f"Error: File '{metadata_path}' not found")
        return

    print("Reading data...")
    
    # 2. Read data
    try:
        # Read links file (columns: movieId, imdbId, tmdbId)
        df_links = pd.read_csv(links_path)
        
        # Read metadata file
        # low_memory=False is used to prevent warnings on large files with mixed types
        df_metadata = pd.read_csv(metadata_path, low_memory=False)
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 3. Preprocess IDs
    if 'tmdbId' not in df_links.columns:
        print("Error: Column 'tmdbId' not found in links file.")
        return
    
    if 'id' not in df_metadata.columns:
        print("Error: Column 'id' not found in metadata file.")
        return

    # Process tmdbId from links.csv
    # tmdbId can contain NaNs (movies without TMDB link), we drop them
    df_links_clean = df_links.dropna(subset=['tmdbId']).copy()
    # Convert to int for comparison
    df_links_clean['tmdbId'] = df_links_clean['tmdbId'].astype(int)
    tmdb_ids = set(df_links_clean['tmdbId'].unique())

    # Process id from movies_metadata.csv
    # Clean metadata IDs: coerce errors to NaN (handles bad data like dates), then drop
    df_metadata['clean_id'] = pd.to_numeric(df_metadata['id'], errors='coerce')
    valid_metadata_ids = set(df_metadata['clean_id'].dropna().astype(int))

    print(f"Statistics:")
    print(f"   - Valid tmdbIds in links.csv: {len(tmdb_ids)}")
    print(f"   - Valid ids in movies_metadata.csv: {len(valid_metadata_ids)}")

    # 4. Find missing IDs
    # IDs present in links.csv but NOT in movies_metadata.csv
    missing_ids = tmdb_ids - valid_metadata_ids

    # 5. Output results
    print("\n" + "="*30)
    if len(missing_ids) == 0:
        print("SUCCESS!")
        print("All tmdbIds from links.csv exist in movies_metadata.csv.")
    else:
        print(f"WARNING: Found {len(missing_ids)} tmdbIds in links.csv that do not exist in movies_metadata.csv!")
        print("List of missing tmdbIds (first 10):")
        print(list(missing_ids)[:10])
        
        # Optional: Save missing IDs to file
        # pd.DataFrame(list(missing_ids), columns=['missing_tmdbId']).to_csv('./ml-20m-psm/data/missing_tmdb_ids.csv', index=False)

if __name__ == "__main__":
    check_tmdb_in_metadata(links_path='./ml-20m-psm/data/links.csv', metadata_path='./ml-20m-psm/data/movies_metadata.csv')