import pandas as pd
import numpy as np

def recall_at_k(actual, predicted, k):
    """Calculate Recall@K"""
    if len(actual) == 0:
        return 0.0
    
    predicted_k = predicted[:k]
    hits = len(set(actual) & set(predicted_k))
    return hits / len(actual)

def ndcg_at_k(actual, predicted, k):
    """Calculate NDCG@K"""
    if len(actual) == 0:
        return 0.0
    
    predicted_k = predicted[:k]
    dcg = 0.0
    for i, item in enumerate(predicted_k):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)
    
    # Ideal DCG
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual), k))])
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_top20(top20_file, test_file, k_values=[5, 10, 20]):
    """
    Đánh giá top 20 movies với test set
    
    Args:
        top20_file: File CSV chứa top 20 movies (có cột movie_id)
        test_file: File CSV test set (userId, movieId, rating, timestamp)
        k_values: Danh sách K để đánh giá
    """
    print(f"\nĐánh giá: {top20_file}")
    print("="*60)
    
    # 1. Đọc top 20 movies
    df_top20 = pd.read_csv(top20_file)
    top20_movies = df_top20['movie_id'].tolist()
    print(f"Top 20 movies: {top20_movies[:5]}... (total: {len(top20_movies)})")
    
    # 2. Đọc test set
    df_test = pd.read_csv(test_file)
    print(f"Test set size: {len(df_test)}")
    
    # 3. Tạo dict: user -> list of movies trong test
    user_test_movies = {}
    for _, row in df_test.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        
        if user_id not in user_test_movies:
            user_test_movies[user_id] = []
        user_test_movies[user_id].append(movie_id)
    
    print(f"Số users trong test: {len(user_test_movies)}")
    
    # 4. Tính metrics cho từng user
    recalls = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}
    
    for user_id, actual_movies in user_test_movies.items():
        # Ground truth = movies của user trong test set
        # Predicted = top 20 movies (giống nhau cho tất cả users)
        
        for k in k_values:
            recall = recall_at_k(actual_movies, top20_movies, k)
            ndcg = ndcg_at_k(actual_movies, top20_movies, k)
            
            recalls[k].append(recall)
            ndcgs[k].append(ndcg)
    
    # 5. Tính trung bình
    results = {}
    for k in k_values:
        results[f'Recall@{k}'] = np.mean(recalls[k])
        results[f'NDCG@{k}'] = np.mean(ndcgs[k])
    
    return results

def print_results(method_name, results):
    """In kết quả"""
    print(f"\n{method_name}")
    print("-"*60)
    print(f"{'Metric':<15} {'K=5':<15} {'K=10':<15} {'K=20':<15}")
    print("-"*60)
    
    r5 = results['Recall@5']
    r10 = results['Recall@10']
    r20 = results['Recall@20']
    print(f"{'Recall':<15} {r5:<15.5f} {r10:<15.5f} {r20:<15.5f}")
    
    n5 = results['NDCG@5']
    n10 = results['NDCG@10']
    n20 = results['NDCG@20']
    print(f"{'NDCG':<15} {n5:<15.5f} {n10:<15.5f} {n20:<15.5f}")

def main():
    print("="*60)
    print("ĐÁNH GIÁ TOP 20 BASELINES VỚI TEST SET")
    print("="*60)
    
    # Đường dẫn files
    test_file = "./ml-20m-psm/data/test.csv"
    
    baselines = {
        "Top 20 by Average Rating": "./tools/top_20_avg_rating.csv",
        "Top 20 by Sum Rating": "./tools/top_20_sum_rating.csv"
    }
    
    all_results = {}
    
    # Đánh giá từng baseline
    for method_name, baseline_file in baselines.items():
        results = evaluate_top20(baseline_file, test_file, k_values=[5, 10, 20])
        all_results[method_name] = results
        print_results(method_name, results)
    
    # Bảng so sánh
    print("\n" + "="*60)
    print("BẢNG SO SÁNH")
    print("="*60)
    
    print(f"\n{'Method':<30} {'Recall@5':<12} {'Recall@10':<12} {'Recall@20':<12}")
    print("-"*66)
    for method, results in all_results.items():
        r5 = results['Recall@5']
        r10 = results['Recall@10']
        r20 = results['Recall@20']
        print(f"{method:<30} {r5:<12.5f} {r10:<12.5f} {r20:<12.5f}")
    
    print(f"\n{'Method':<30} {'NDCG@5':<12} {'NDCG@10':<12} {'NDCG@20':<12}")
    print("-"*66)
    for method, results in all_results.items():
        n5 = results['NDCG@5']
        n10 = results['NDCG@10']
        n20 = results['NDCG@20']
        print(f"{method:<30} {n5:<12.5f} {n10:<12.5f} {n20:<12.5f}")

if __name__ == "__main__":
    main()