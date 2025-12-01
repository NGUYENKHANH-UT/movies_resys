import pandas as pd
import os

def filter_ratings_by_blacklist(
    ratings_path='./ml-20m-psm/ratings.csv', 
    invalid_ratings_path='./ml-20m-psm/data/invalid_ratings.csv', 
    output_path='./ml-20m-psm/data/ratings.csv'
):
    """
    Reads the main ratings file and the invalid ratings file.
    Removes any ratings found in the invalid list from the main list.
    """
    
    # 1. Check if input files exist
    if not os.path.exists(ratings_path):
        print(f"Error: File '{ratings_path}' not found")
        return
    if not os.path.exists(invalid_ratings_path):
        print(f"Error: File '{invalid_ratings_path}' not found")
        return

    print("Reading data...")
    
    try:
        # Read the main ratings file
        # Assumes columns: userId, movieId, rating, timestamp
        df_ratings = pd.read_csv(ratings_path)
        
        # Read the invalid ratings file (the blacklist)
        df_invalid = pd.read_csv(invalid_ratings_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 2. Identify Invalid Movie IDs
    # Since the invalidity is based on movieId not existing, we can simply 
    # collect all unique movieIds from the invalid_ratings.csv
    if 'movieId' not in df_invalid.columns:
        print("Error: Column 'movieId' not found in invalid_ratings.csv")
        return

    invalid_movie_ids = set(df_invalid['movieId'].unique())
    
    original_count = len(df_ratings)
    print(f"Original ratings count: {original_count}")
    print(f"Number of invalid movies to remove: {len(invalid_movie_ids)}")

    # 3. Filter ratings
    # We keep rows where movieId is NOT in the invalid_movie_ids set
    # The tilde (~) operator negates the condition (isIn -> isNotIn)
    print("Filtering ratings based on invalid list...")
    df_clean = df_ratings[~df_ratings['movieId'].isin(invalid_movie_ids)]

    cleaned_count = len(df_clean)
    removed_count = original_count - cleaned_count

    print(f"Final ratings count: {cleaned_count}")
    print(f"Removed {removed_count} ratings.")

    # 4. Save to output file
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df_clean.to_csv(output_path, index=False)
        print(f"Successfully saved clean ratings to '{output_path}'.")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    filter_ratings_by_blacklist()