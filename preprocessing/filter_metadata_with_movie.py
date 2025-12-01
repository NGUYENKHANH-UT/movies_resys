import pandas as pd
import os

def filter_metadata_by_tmdb(
    links_path='./ml-20m-psm/data/links.csv',
    metadata_path='./ml-20m-psm/movies_metadata.csv',
    output_path='./ml-20m-psm/data/movies_metadata.csv'
):
    """
    Filters movies_metadata.csv to keep only rows where the 'id' exists 
    in the 'tmdbId' column of links.csv.
    """

    # 1. Check if input files exist
    if not os.path.exists(links_path):
        print(f"Error: File '{links_path}' not found.")
        return
    if not os.path.exists(metadata_path):
        print(f"Error: File '{metadata_path}' not found.")
        return

    print("Reading data files...")

    try:
        # 2. Process links.csv (Source of Truth)
        # Read links file
        df_links = pd.read_csv(links_path)
        
        if 'tmdbId' not in df_links.columns:
            print("Error: Column 'tmdbId' not found in links.csv")
            return

        # Remove rows with empty tmdbId and convert to integer
        # tmdbId in links.csv can be float (e.g. 862.0), we need int for comparison
        valid_tmdb_ids = set(df_links['tmdbId'].dropna().astype(int))
        
        print(f"Loaded {len(valid_tmdb_ids)} valid tmdbIds from links.csv")

        # 3. Process movies_metadata.csv (Target to filter)
        # low_memory=False helps with mixed types in large files
        df_metadata = pd.read_csv(metadata_path, low_memory=False)
        
        if 'id' not in df_metadata.columns:
            print("Error: Column 'id' not found in metadata file")
            return

        original_count = len(df_metadata)
        print(f"Original metadata rows: {original_count}")

    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # 4. Clean Metadata IDs
    # Metadata 'id' column often contains dirty data (dates, strings).
    # We create a temporary column 'clean_id' to handle this safely.
    # errors='coerce' turns non-numeric values into NaN
    df_metadata['clean_id'] = pd.to_numeric(df_metadata['id'], errors='coerce')

    # Drop rows where ID could not be converted to a number
    df_metadata_clean = df_metadata.dropna(subset=['clean_id'])
    
    # Convert to integer for matching
    df_metadata_clean['clean_id'] = df_metadata_clean['clean_id'].astype(int)

    # 5. Filter Data
    # Keep row if 'clean_id' is in 'valid_tmdb_ids'
    print("Filtering metadata...")
    df_filtered = df_metadata_clean[df_metadata_clean['clean_id'].isin(valid_tmdb_ids)]

    # Remove the temporary column before saving
    df_filtered = df_filtered.drop(columns=['clean_id'])

    filtered_count = len(df_filtered)
    print(f"Filtered metadata rows: {filtered_count}")
    print(f"Removed {original_count - filtered_count} rows.")

    # 6. Save output
    try:
        df_filtered.to_csv(output_path, index=False)
        print(f"Successfully saved filtered metadata to '{output_path}'.")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    filter_metadata_by_tmdb()