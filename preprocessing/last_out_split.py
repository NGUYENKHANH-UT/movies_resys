import csv
import os
import collections
from tqdm import tqdm

def split_ratings_last_out(input_file='ratings.csv', output_dir='.'):
    """
    Chia file ratings.csv thành train/valid/test theo chiến lược last-out.
    Logic tham khảo từ file last_out_split.py.
    """
    
    # Check input file
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    print(f"Đang đọc dữ liệu từ {input_file}...")
    
    # 1. Read data
    inters = []
    header = None
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("File rỗng!")
            return

        for row in tqdm(reader, desc="Loading CSV"):
            # row: [userId, movieId, rating, timestamp]
            try:
                user_id = row[0]
                item_id = row[1]
                rating = row[2]
                timestamp = int(row[3])
                inters.append((user_id, item_id, rating, timestamp))
            except ValueError:
                continue 

    print("Inprocessing data...")
    
    user2inters = collections.defaultdict(list)
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append(inter)

    train_data = []
    valid_data = []
    test_data = []

    stats = {
        'processed_users': 0,
        'train_items': 0,
        'valid_items': 0,
        'test_items': 0
    }

    for user in tqdm(user2inters, desc="Splitting"):
        user_inters = user2inters[user]
        
        user_inters.sort(key=lambda x: x[3])
        
        his_items = set()
        clean_inters = []
        for inter in user_inters:
            item = inter[1]
            if item in his_items:
                continue
            his_items.add(item)
            clean_inters.append(inter)
        
        cur_inter = clean_inters
        
        if not cur_inter:
            continue

        test_data.append(cur_inter[-1])
        stats['test_items'] += 1
        
        if len(cur_inter) > 1:
            valid_data.append(cur_inter[-2])
            stats['valid_items'] += 1
            
        if len(cur_inter) > 2:
            train_segment = cur_inter[:-2]
            train_data.extend(train_segment)
            stats['train_items'] += len(train_segment)
            
        stats['processed_users'] += 1

    def write_csv(filename, data_list):
        filepath = os.path.join(output_dir, filename)
        print(f"Đang ghi file {filename} ({len(data_list)} dòng)...")
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(data_list)

    write_csv('train.csv', train_data)
    write_csv('valid.csv', valid_data)
    write_csv('test.csv', test_data)

    print("\n--- Hoàn tất ---")
    print(f"Sum of User: {stats['processed_users']}")
    print(f"Train: {stats['train_items']}")
    print(f"Valid: {stats['valid_items']}")
    print(f"Test : {stats['test_items']}")

if __name__ == '__main__':
    file_path = "./ml-20m-psm/data/ratings.csv"
    output_dir = "./ml-20m-psm/data/"
    split_ratings_last_out(input_file=file_path, output_dir=output_dir)