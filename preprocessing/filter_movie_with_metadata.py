import pandas as pd
import os

def filter_metadata(
    missing_ids_path='./ml-20m-psm/data/missing_tmdb_ids.csv', 
    links_path='./ml-20m-psm/data/links.csv', 
    movies_path='./ml-20m-psm/data/movies.csv',
    output_links='./ml-20m-psm/data/links_1.csv',
    output_movies='./ml-20m-psm/data/movies_1.csv'
):
    """
    Filters links.csv to remove rows with missing metadata (based on missing_tmdb_ids.csv).
    Then filters movies.csv to keep only movies present in the filtered links.
    """
    
    # 1. Check if input files exist
    if not os.path.exists(missing_ids_path):
        print(f"Error: File '{missing_ids_path}' not found.")
        return
    if not os.path.exists(links_path):
        print(f"Error: File '{links_path}' not found.")
        return
    if not os.path.exists(movies_path):
        print(f"Error: File '{movies_path}' not found.")
        return

    print("Reading data files...")

    try:
        # Load missing TMDB IDs
        # Assuming the file has a header like 'missing_tmdbId'
        df_missing = pd.read_csv(missing_ids_path)
        # Flatten to a set for faster lookup
        missing_tmdb_ids = set(df_missing.iloc[:, 0].dropna().astype(int))
        
        # Load links.csv
        df_links = pd.read_csv(links_path)
        
        # Load movies.csv
        # Assumes header exists: movieId, title, genres
        df_movies = pd.read_csv(movies_path)

    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    print(f"Original links count: {len(df_links)}")
    print(f"Original movies count: {len(df_movies)}")

    # ---------------------------------------------------------
    # STEP 1: Filter links.csv
    # ---------------------------------------------------------
    print("Filtering links...")
    
    # Drop rows where tmdbId is NaN first, as they cannot be checked against missing list
    df_links_clean = df_links.dropna(subset=['tmdbId']).copy()
    
    # Ensure tmdbId is int for comparison
    df_links_clean['tmdbId'] = df_links_clean['tmdbId'].astype(int)

    # Keep rows where tmdbId is NOT in the missing set
    # The tilde (~) operator negates the condition
    df_links_filtered = df_links_clean[~df_links_clean['tmdbId'].isin(missing_tmdb_ids)]

    # Get the list of valid movieIds from the filtered links
    valid_movie_ids = set(df_links_filtered['movieId'].unique())

    print(f"Filtered links count: {len(df_links_filtered)}")
    print(f"Number of valid movies: {len(valid_movie_ids)}")

    # ---------------------------------------------------------
    # STEP 2: Filter movies.csv
    # ---------------------------------------------------------
    print("Filtering movies...")

    # Keep movies only if movieId exists in the valid_movie_ids set (derived from links)
    if 'movieId' in df_movies.columns:
        df_movies_filtered = df_movies[df_movies['movieId'].isin(valid_movie_ids)]
        print(f"Filtered movies count: {len(df_movies_filtered)}")
    else:
        print("Error: 'movieId' column not found in movies.csv. Skipping filter.")
        df_movies_filtered = pd.DataFrame() # Empty DF

    # ---------------------------------------------------------
    # STEP 3: Save output files
    # ---------------------------------------------------------
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_links), exist_ok=True)
        
        df_links_filtered.to_csv(output_links, index=False)
        
        if not df_movies_filtered.empty:
            df_movies_filtered.to_csv(output_movies, index=False)
            print(f"Successfully saved '{output_links}' and '{output_movies}'.")
        else:
            print(f"Successfully saved '{output_links}'. Movies file was not saved due to errors.")
            
    except Exception as e:
        print(f"Error saving output files: {e}")

if __name__ == "__main__":
    filter_metadata()