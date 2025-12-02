import pandas as pd

def get_top_users_data(input_path, output_path, target_rows=2000000):
    print("Đang đọc file...")
    # Đọc file (giả sử file chưa được sort, nếu sort rồi thì vẫn chạy đúng)
    df = pd.read_csv(input_path)
    
    # Lấy danh sách user theo thứ tự xuất hiện (hoặc sort nếu muốn)
    # unique() của pandas giữ nguyên thứ tự xuất hiện
    users = df['userId'].unique()
    
    selected_users = []
    current_count = 0
    
    # Lấy user từ đầu danh sách cho đến khi đủ số dòng
    for u in users:
        # Đếm số dòng của user này
        # (Cách này hơi chậm nếu loop nhiều, nhưng với 100k dòng thì rất nhanh)
        count = len(df[df['userId'] == u])
        selected_users.append(u)
        current_count += count
        
        if current_count >= target_rows:
            break
            
    print(f"Đã lấy {len(selected_users)} users đầu tiên. Tổng số dòng: {current_count}")
    
    # Lọc ra dataframe cuối cùng
    df_small = df[df['userId'].isin(selected_users)]
    
    df_small.to_csv(output_path, index=False)
    print("Xong!")

get_top_users_data('./ml-20m-psm/data/ratings.csv', './ml-20m-psm/data/ratings.csv')