import pandas as pd
import os

def check_movie_id_consistency(ratings_path='ratings.csv', movies_path='movies.csv'):
    """
    Checks if all movieIds in the ratings file exist in the movies file.
    """
    
    # 1. Check if files exist
    if not os.path.exists(ratings_path):
        print(f"Error: File '{ratings_path}' not found")
        return
    if not os.path.exists(movies_path):
        print(f"Error: File '{movies_path}' not found")
        return

    print("Reading data...")
    
    # 2. Read data
    try:
        # Read ratings file (assumes header exists based on your sample data)
        df_ratings = pd.read_csv(ratings_path)
        
    except Exception as e:
        print(f"Error reading CSV file rating : {e}")
        return
    
    try:  
        # Read movies file
        df_movies = pd.read_csv(movies_path)
    except Exception as e:
        print(f"Error reading CSV file movies : {e}")
        return

    # 3. Get set of unique IDs
    # Ensure the column name matches your CSV file (e.g., 'movieId')
    if 'movieId' not in df_ratings.columns or 'movieId' not in df_movies.columns:
        print("Error: Column 'movieId' not found in one of the files.")
        return

    ratings_movie_ids = set(df_ratings['movieId'].unique())
    movies_movie_ids = set(df_movies['movieId'].unique())

    print(f"Statistics:")
    print(f"   - Count in ratings: {len(ratings_movie_ids)}")
    print(f"   - Count in movies:  {len(movies_movie_ids)}")

    # 4. Find IDs present in ratings but NOT in movies
    missing_ids = ratings_movie_ids - movies_movie_ids

    # 5. Output results
    print("\n" + "="*30)
    if len(missing_ids) == 0:
        print("SUCCESS!")
        print("All movieIds in ratings are valid and exist in movies.csv.")
    else:
        print(f"WARNING: Found {len(missing_ids)} movieIds in ratings that do not exist in movies.csv!")
        print("List of missing IDs (first 10):")
        print(list(missing_ids)[:10])
        
        # Optional: Save invalid rows to a file
        # invalid_rows = df_ratings[df_ratings['movieId'].isin(missing_ids)]
        # invalid_rows.to_csv('./ml-20m-psm/data/invalid_ratings.csv', index=False)

if __name__ == "__main__":
    check_movie_id_consistency(ratings_path='./ml-20m-psm/data/ratings.csv', movies_path='./ml-20m-psm/data/movies.csv')