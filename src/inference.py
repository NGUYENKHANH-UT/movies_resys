import time
import json
import os
import sys
import numpy as np
import redis
import torch
from PIL import Image
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# Add parent directory to path to import modules from src
# Adjust this depending on where you place the script (root or tools/)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import MargoDataset

class MargoInferenceEngine:
    def __init__(self):
        print("Initializing MARGO Inference Engine...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Connect Databases
        self._connect_redis()
        self._connect_milvus()
        
        # 2. Load Models
        # In a real production environment, lazy loading or serving these as separate microservices is recommended
        print("Loading Encoders (SBERT & CLIP)...")
        self.sbert = SentenceTransformer('all-mpnet-base-v2', device=self.device)
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
        # 3. Load User History (for Masking/Filtering)
        # We load the dataset to build a map of user history to filter out seen movies.
        # Ideally, this history should also be stored in Redis for faster access.
        print("Loading User History Map...")
        self.dataset = MargoDataset()
        self.user_history_map = self._build_history_map()
        
        print("Engine Ready.")

    def _connect_redis(self):
        """
        Connect to Upstash Redis to fetch cached User Embeddings.
        """
        try:
            self.redis = redis.Redis(
                host=Config.redis_host,
                port=Config.redis_port,
                password=Config.redis_password,
                ssl=Config.redis_ssl,
                decode_responses=True
            )
            self.redis.ping()
            print(f" -> Connected to Redis: {Config.redis_host}")
        except Exception as e:
            print(f" -> Redis Connection Error: {e}")
            sys.exit(1)

    def _connect_milvus(self):
        """
        Connect to Milvus/Zilliz to access Item Collections.
        """
        try:
            connections.connect("default", uri=Config.milvus_uri, token=Config.milvus_token)
            
            # Collection 1: Raw Features (Used for Text/Image Search)
            self.col_raw = Collection("movies_multimodal")
            self.col_raw.load()
            
            # Collection 2: Final Embeddings (Used for Personalized Re-ranking)
            self.col_final = Collection("movies_margo_final")
            self.col_final.load()
            
            print(" -> Connected to Milvus & Collections Loaded")
        except Exception as e:
            print(f" -> Milvus Connection Error: {e}")
            sys.exit(1)

    def _build_history_map(self):
        """
        Build a dictionary mapping Internal User ID to a Set of Original Movie IDs (watched items).
        Used to filter out items the user has already interacted with.
        """
        history = {}
        # edge_index is a tensor [2, Num_Edges] containing (User, Item) pairs
        u_indices = self.dataset.edge_index[0].cpu().numpy()
        i_indices = self.dataset.edge_index[1].cpu().numpy()
        
        num_users = self.dataset.num_users
        
        for u, i in zip(u_indices, i_indices):
            # In the graph, item_node_id = item_internal_id + num_users
            if u < num_users and i >= num_users:
                item_internal_id = i - num_users
                
                # Map Internal ID to Original Movie ID
                if item_internal_id in self.dataset.id2item:
                    movie_id = self.dataset.id2item[item_internal_id]
                    if u not in history: history[u] = set()
                    history[u].add(movie_id)
        return history

    def _get_user_vector(self, user_id):
        """Fetch 128-dim user embedding from Redis."""
        key = f"user:{user_id}"
        vec_str = self.redis.get(key)
        if vec_str:
            return json.loads(vec_str)
        return None

    def _encode_text_sbert(self, text):
        """Encode text using SBERT (Semantic Search)."""
        return self.sbert.encode(text).tolist()

    def _encode_text_clip(self, text):
        """Encode text using CLIP Text Encoder (Visual Description)."""
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()

    def _encode_image_clip(self, image_path):
        """Encode image using CLIP Image Encoder (Visual Similarity)."""
        try:
            image = Image.open(image_path)
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
            return features.cpu().numpy()[0].tolist()
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def recommend(self, user_internal_id, query_text=None, query_image_path=None, top_k=10):
        """
        Main function to get recommendations.
        
        Args:
            user_internal_id (int): The internal ID of the user.
            query_text (str): Optional text query (e.g., "superhero movie").
            query_image_path (str): Optional path to an image file for query.
            top_k (int): Number of items to return.
            
        Returns:
            List of dictionaries containing 'movie_id' and 'score'.
        """
        start_time = time.time()
        
        # 1. Fetch User Vector from Redis
        user_vec = self._get_user_vector(user_internal_id)
        if not user_vec:
            print(f"Warning: User ID {user_internal_id} not found in Cache. Treating as cold user (vector=0).")
            # In a real scenario, you might want to return popular items here.
            # For now, we initialize a zero vector to allow the code to proceed.
            user_vec = [0.0] * 128 
            
        # 2. Determine Candidate Pool
        candidate_ids = set()
        
        # CASE 1: Personalized Search (Query Provided)
        if query_text or query_image_path:
            print(f"Mode: Search (Text: '{query_text}', Img: {query_image_path})")
            
            # A. Semantic Search (SBERT -> Text Embedding)
            # Goal: Match semantic meaning (plot, genre, etc.)
            if query_text:
                vec_sbert = self._encode_text_sbert(query_text)
                res = self.col_raw.search(
                    data=[vec_sbert], 
                    anns_field="text_emb", 
                    param={"metric_type": "IP", "params": {"nprobe": 10}}, 
                    limit=100, 
                    output_fields=["movie_id"]
                )
                candidate_ids.update([h.id for h in res[0]])
                
            # B. Visual Description (CLIP Text -> Visual Embedding)
            # Goal: Match visual description (e.g., "dark poster", "animation")
            if query_text:
                vec_clip_txt = self._encode_text_clip(query_text)
                res = self.col_raw.search(
                    data=[vec_clip_txt], 
                    anns_field="visual_emb", 
                    param={"metric_type": "IP", "params": {"nprobe": 10}}, 
                    limit=40, 
                    output_fields=["movie_id"]
                )
                candidate_ids.update([h.id for h in res[0]])
                
            # C. Visual Similarity (CLIP Image -> Visual Embedding)
            # Goal: Find visually similar items to the uploaded image
            if query_image_path:
                vec_clip_img = self._encode_image_clip(query_image_path)
                if vec_clip_img:
                    res = self.col_raw.search(
                        data=[vec_clip_img], 
                        anns_field="visual_emb", 
                        param={"metric_type": "IP", "params": {"nprobe": 10}}, 
                        limit=60, 
                        output_fields=["movie_id"]
                    )
                    candidate_ids.update([h.id for h in res[0]])
            
            # Convert set to list for query
            candidate_ids = list(candidate_ids)
            print(f" -> Found {len(candidate_ids)} candidates from Query.")
            
            # Re-rank the candidates using User Interest
            final_results = self._rerank_candidates(user_vec, candidate_ids, user_internal_id, top_k)

        # CASE 2: Pure Recommendation (No Query)
        else:
            print("Mode: Personalized Recommendation")
            
            # Search directly in the Final Collection using the User Vector
            # Retrieve more items than k to allow for filtering watched items
            seen_movies = self.user_history_map.get(user_internal_id, set())
            search_limit = top_k + len(seen_movies) + 50
            
            res = self.col_final.search(
                data=[user_vec], 
                anns_field="fused_emb", 
                param={"metric_type": "IP", "params": {"nprobe": 10}}, 
                limit=search_limit,
                output_fields=["movie_id"]
            )
            
            final_results = []
            
            for hit in res[0]:
                if hit.id not in seen_movies:
                    final_results.append({"movie_id": hit.id, "score": hit.score})
                    if len(final_results) >= top_k:
                        break

        print(f"Time taken: {time.time() - start_time:.4f}s")
        return final_results

    def _rerank_candidates(self, user_vec, candidate_ids, user_id, top_k):
        """
        Re-rank a list of candidate items based on the user's MARGO score.
        Score = DotProduct(User_Vector, Item_Fused_Vector)
        """
        if not candidate_ids:
            return []
            
        # 1. Fetch Fused Embeddings from Milvus Final Collection
        # Format the list of IDs for the query expression
        str_ids = str(candidate_ids).replace('[', '').replace(']', '')
        
        # Note: Milvus has a limit on query length. If candidate_ids is very large,
        # consider splitting this into batches.
        res = self.col_final.query(
            expr=f"movie_id in [{str_ids}]", 
            output_fields=["movie_id", "fused_emb"]
        )
        
        # 2. Calculate Scores and Filter Watched Items
        seen_movies = self.user_history_map.get(user_id, set())
        scores = []
        
        user_vec_np = np.array(user_vec)
        
        for item in res:
            m_id = item['movie_id']
            if m_id in seen_movies:
                continue
                
            item_vec = np.array(item['fused_emb'])
            
            # Calculate MARGO score (Inner Product)
            score = np.dot(user_vec_np, item_vec)
            scores.append({"movie_id": m_id, "score": float(score)})
            
        # 3. Sort by score descending and take top k
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]

# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    engine = MargoInferenceEngine()
    
    # Example User ID (Internal ID from 0 to Num_Users-1)
    TEST_USER_ID = 10
    
    print("\n--- TEST 1: Pure Recommendation (User ID Only) ---")
    recs = engine.recommend(TEST_USER_ID, top_k=5)
    for r in recs: print(r)
    
    print("\n--- TEST 2: Search + Personalization (Query: 'superhero action movie') ---")
    recs = engine.recommend(TEST_USER_ID, query_text="superhero action movie", top_k=5)
    for r in recs: print(r)
    
    # print("\n--- TEST 3: Image Query + Personalization ---")
    # recs = engine.recommend(TEST_USER_ID, query_image_path="./test_poster.jpg", top_k=5)
    # for r in recs: print(r)