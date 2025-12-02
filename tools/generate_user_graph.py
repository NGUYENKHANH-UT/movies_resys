import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
import os

# --- Configuration Stub for standalone run ---
# In a real environment, replace this with imports or adjust paths accordingly.
class ConfigStub:
    base_dir = './ml-20m-psm'
    ratings_path = os.path.join(base_dir, 'data/ratings.csv')
    user_graph_path = os.path.join(base_dir, 'data/user_graph_dict.npy')
    K_UCG = 10 # Top-K neighbors for User Co-occurrence Graph

def generate_user_co_occurrence_graph(ratings_path, output_path, k):
    """
    Generates the User Co-occurrence Graph (UCG) matrix and saves the top-k neighbors
    and their edge weights (co-occurrence count) into a dictionary.
    """
    if not os.path.exists(ratings_path):
        print(f"Error: Ratings file not found at {ratings_path}")
        return

    print(f"Loading data from {ratings_path}...")
    df = pd.read_csv(ratings_path)
    
    # 1. Map original IDs to internal IDs [0, num_users - 1]
    unique_users = sorted(df['userId'].unique())
    user_to_id = {u: i for i, u in enumerate(unique_users)}
    num_user = len(unique_users)
    
    # Filter for unique (userId, movieId) pairs
    df_interactions = df[['userId', 'movieId']].drop_duplicates()
    df_interactions['internal_userId'] = df_interactions['userId'].map(user_to_id)
    
    # 2. Build adjacency dictionary (User internal ID -> Set of Item original IDs)
    item_by_user = defaultdict(set)
    for index, row in tqdm(df_interactions.iterrows(), total=len(df_interactions), desc="Building Item Sets"):
        item_by_user[row['internal_userId']].add(row['movieId'])

    # 3. Compute Co-occurrence Matrix
    user_graph_matrix = np.zeros((num_user, num_user), dtype=np.int32)
    
    print("Computing User Co-occurrence Matrix...")
    for i in tqdm(range(num_user), desc="Calculating Co-occurrence"):
        for j in range(i + 1, num_user):
            co_occurrence_count = len(item_by_user[i].intersection(item_by_user[j]))
            if co_occurrence_count > 0:
                user_graph_matrix[i, j] = co_occurrence_count
                user_graph_matrix[j, i] = co_occurrence_count
                
    # 4. Extract Top-K Neighbors and Weights for Dict
    user_graph_dict = {}
    print(f"Extracting Top-{k} Neighbors...")
    for i in tqdm(range(num_user), desc="Extracting Top-K"):
        user_i_row = user_graph_matrix[i, :]
        
        # Get top-k indices/values and sort descendingly
        # This is a robust way to get the top-k indices
        top_k_indices = np.argsort(user_i_row)[::-1][:k]
        
        # Filter out zero weights and self-loop if necessary
        valid_neighbors_mask = user_i_row[top_k_indices] > 0
        final_indices = top_k_indices[valid_neighbors_mask]
        final_values = user_i_row[final_indices]
        
        # Store as [indices, values]
        user_graph_dict[i] = [final_indices.tolist(), final_values.tolist()]
        
    # 5. Save output
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, user_graph_dict, allow_pickle=True)
    print(f"\nSuccessfully saved User Graph Dictionary to {output_path}")

if __name__ == "__main__":
    # --- Instructions: Run this file first ---
    # Assuming the config is located one level up and data exists.
    # Replace ConfigStub with your actual config import if running from another location.
    generate_user_co_occurrence_graph(ConfigStub.ratings_path, ConfigStub.user_graph_path, ConfigStub.K_UCG)