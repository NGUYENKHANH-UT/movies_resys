import pandas as pd
import os

def check_and_filter_duplicates(
    metadata_path='./ml-20m-psm/data/movies_metadata_clean.csv', 
    output_duplicates_path='./ml-20m-psm/data/metadata_duplicates.csv',
    output_clean_path='./ml-20m-psm/data/movies_metadata_clean.csv'
):
    """
    Checks for duplicate IDs in movies_metadata.csv.
    Saves duplicates to a separate file for review.
    Saves a clean version (duplicates removed) to a new file.
    """
    
    # 1. Check if input file exists
    if not os.path.exists(metadata_path):
        print(f"Error: File '{metadata_path}' not found.")
        return

    print("Reading metadata file...")
    
    try:
        # Read metadata file
        # low_memory=False handles mixed data types in columns
        df = pd.read_csv(metadata_path, low_memory=False)
        
        if 'id' not in df.columns:
            print("Error: Column 'id' not found in metadata file.")
            return
            
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    original_count = len(df)
    print(f"Original rows: {original_count}")

    # 2. Clean IDs
    # The 'id' column in this dataset is notorious for having bad data (dates, strings)
    # We convert to numeric, coercing errors to NaN
    print("Cleaning IDs...")
    df['clean_id'] = pd.to_numeric(df['id'], errors='coerce')
    
    # Drop rows where ID is invalid (NaN)
    df_valid = df.dropna(subset=['clean_id']).copy()
    
    # Convert to integer for accurate duplicate checking
    df_valid['clean_id'] = df_valid['clean_id'].astype(int)
    
    valid_count = len(df_valid)
    print(f"Rows with valid numeric IDs: {valid_count}")
    print(f"Dropped {original_count - valid_count} rows with invalid IDs.")

    # 3. Check for duplicates
    # duplicated() returns True for items that appear more than once
    # keep=False marks all duplicates as True
    duplicate_mask = df_valid.duplicated(subset=['clean_id'], keep=False)
    
    df_duplicates = df_valid[duplicate_mask]
    num_duplicates = len(df_duplicates)
    
    print("-" * 30)
    if num_duplicates == 0:
        print("SUCCESS: No duplicate IDs found among valid rows.")
    else:
        print(f"WARNING: Found {num_duplicates} rows involved in duplication.")
        
        # Count unique IDs that have duplicates
        num_unique_duplicate_ids = df_duplicates['clean_id'].nunique()
        print(f"Number of unique IDs that are duplicated: {num_unique_duplicate_ids}")
        
        # Save duplicates for review
        try:
            # Drop the temporary clean_id column before saving if you want original format,
            # or keep it. Here we remove it to match original structure.
            df_duplicates_save = df_duplicates.drop(columns=['clean_id'])
            df_duplicates_save.to_csv(output_duplicates_path, index=False)
            print(f"Duplicate rows saved to '{output_duplicates_path}'")
        except Exception as e:
            print(f"Error saving duplicates log: {e}")

    # 4. Filter (Remove duplicates)
    # keep='first' keeps the first occurrence and drops subsequent ones
    print("-" * 30)
    print("Filtering duplicates...")
    
    df_clean = df_valid.drop_duplicates(subset=['clean_id'], keep='first')
    
    # Remove the temporary column
    df_clean = df_clean.drop(columns=['clean_id'])
    
    final_count = len(df_clean)
    print(f"Final clean rows: {final_count}")
    
    # 5. Save clean file
    """try:
        df_clean.to_csv(output_clean_path, index=False)
        print(f"Successfully saved clean metadata to '{output_clean_path}'")
    except Exception as e:
        print(f"Error saving clean file: {e}")"""
    

if __name__ == "__main__":
    check_and_filter_duplicates()