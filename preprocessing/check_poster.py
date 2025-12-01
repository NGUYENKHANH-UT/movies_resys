import pandas as pd
import os

def check_missing_posters(
    movies_path='./ml-20m-psm/data/movies_1.csv', 
    posters_dir='./ml-20m-psm/posters', 
    # output_log='./ml-20m-psm/data/missing_posters.csv'
):
    """
    Checks which movies from movies.csv do not have a corresponding image in the posters folder.
    Logs missing movieIds to a CSV file.
    """
    
    # 1. Check if movies file exists
    if not os.path.exists(movies_path):
        print(f"Error: File '{movies_path}' not found.")
        return

    # 2. Check if posters directory exists
    if not os.path.exists(posters_dir):
        print(f"Error: Directory '{posters_dir}' not found.")
        return

    print("Reading movies data...")
    try:
        # Read movies.csv (assumes header exists: movieId, title, genres)
        df_movies = pd.read_csv(movies_path)
        
        if 'movieId' not in df_movies.columns:
            print("Error: Column 'movieId' not found in movies.csv")
            return
            
        # Get all movie IDs
        movie_ids = df_movies['movieId'].unique()
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print(f"Total movies to check: {len(movie_ids)}")
    print("Scanning posters directory...")

    # 3. Check for missing posters
    missing_posters = []
    
    # Get set of existing files for faster lookup (optional optimization)
    # existing_files = set(os.listdir(posters_dir))

    for m_id in movie_ids:
        # Construct expected filename: <movieId>.jpg
        image_name = f"{m_id}.jpg"
        image_path = os.path.join(posters_dir, image_name)
        
        # Check if file exists
        if not os.path.isfile(image_path):
            missing_posters.append(m_id)

    # 4. Log results
    print("\n" + "="*30)
    if len(missing_posters) == 0:
        print("SUCCESS!")
        print("All movies have corresponding posters.")
    else:
        print(f"WARNING: Found {len(missing_posters)} movies without posters.")
        
        try:
            # Create a DataFrame for missing IDs
            df_missing = pd.DataFrame(missing_posters, columns=['movieId'])
            
            # Optionally, you can join with original data to see titles of missing movies
            # df_missing = df_missing.merge(df_movies[['movieId', 'title']], on='movieId', how='left')
            
            # Save to CSV
            # df_missing.to_csv(output_log, index=False)
            # print(f"Missing movie IDs logged to '{output_log}'.")
            
            # Print first 5 missing IDs as example
            print("First 5 missing IDs:", missing_posters[:5])
            
        except Exception as e:
            print(f"Error writing log file: {e}")

if __name__ == "__main__":
    # Ensure you have 'movies.csv' and a folder named 'posters' in the same directory
    check_missing_posters()