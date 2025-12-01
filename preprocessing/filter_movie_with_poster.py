import pandas as pd
import os

def filter_all_data(
    missing_posters_path='./ml-20m-psm/data/missing_posters.csv', 
    links_path='./ml-20m-psm/links.csv', 
    movies_path='./ml-20m-psm/movies.csv',
    output_links='./ml-20m-psm/data/links.csv',
    output_movies='./ml-20m-psm/data/movies.csv'
):
    """
    Filters both links.csv and movies.csv to exclude movies listed in missing_posters.csv.
    Saves the results to new CSV files.
    """
    
    # 1. Check if input files exist
    if not os.path.exists(missing_posters_path):
        print(f"Error: File '{missing_posters_path}' not found.")
        return
    if not os.path.exists(links_path):
        print(f"Error: File '{links_path}' not found.")
        return
    if not os.path.exists(movies_path):
        print(f"Error: File '{movies_path}' not found.")
        return

    print("Reading data files...")

    try:
        # 2. Read missing posters list
        # Assuming missing_posters.csv has a header 'movieId'
        df_missing = pd.read_csv(missing_posters_path)
        
        if 'movieId' not in df_missing.columns:
             print("Error: Column 'movieId' not found in missing_posters.csv")
             return
             
        # Create a set of IDs to remove for faster lookup
        missing_ids = set(df_missing['movieId'].unique())
        
        print(f"Total movies to remove (missing posters): {len(missing_ids)}")

    except Exception as e:
        print(f"Error reading missing posters file: {e}")
        return

    # ---------------------------------------------------------
    # PART 1: Filter links.csv
    # ---------------------------------------------------------
    print("-" * 30)
    print("Processing links.csv...")
    try:
        df_links = pd.read_csv(links_path)
        
        if 'movieId' in df_links.columns:
            original_links_count = len(df_links)
            
            # Filter links
            # Keep rows where movieId is NOT in the missing_ids set
            df_links_filtered = df_links[~df_links['movieId'].isin(missing_ids)]
            
            filtered_links_count = len(df_links_filtered)
            print(f"Original links: {original_links_count}")
            print(f"Filtered links: {filtered_links_count}")
            print(f"Removed: {original_links_count - filtered_links_count}")
            
            # Save filtered links
            df_links_filtered.to_csv(output_links, index=False)
            print(f"Saved to '{output_links}'")
        else:
            print("Error: Column 'movieId' not found in links.csv")

    except Exception as e:
        print(f"Error processing links.csv: {e}")

    # ---------------------------------------------------------
    # PART 2: Filter movies.csv
    # ---------------------------------------------------------
    print("-" * 30)
    print("Processing movies.csv...")
    try:
        df_movies = pd.read_csv(movies_path)
        
        if 'movieId' in df_movies.columns:
            original_movies_count = len(df_movies)
            
            # Filter movies
            # Keep rows where movieId is NOT in the missing_ids set
            df_movies_filtered = df_movies[~df_movies['movieId'].isin(missing_ids)]
            
            filtered_movies_count = len(df_movies_filtered)
            print(f"Original movies: {original_movies_count}")
            print(f"Filtered movies: {filtered_movies_count}")
            print(f"Removed: {original_movies_count - filtered_movies_count}")
            
            # Save filtered movies
            df_movies_filtered.to_csv(output_movies, index=False)
            print(f"Saved to '{output_movies}'")
        else:
            print("Error: Column 'movieId' not found in movies.csv")

    except Exception as e:
        print(f"Error processing movies.csv: {e}")

    print("-" * 30)
    print("Done.")

if __name__ == "__main__":
    filter_all_data()