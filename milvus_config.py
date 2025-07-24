from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import json
import numpy as np
import time

# Milvus configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "movie_vectors"

def connect_milvus(max_retries: int = 3):
    """Kết nối tới Milvus server với retry logic"""
    for attempt in range(max_retries):
        try:
            # Disconnect existing connections first
            try:
                connections.disconnect("default")
            except:
                pass
                
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            print(f"Connected to Milvus successfully! (attempt {attempt + 1})")
            return True
        except Exception as e:
            print(f"Failed to connect to Milvus (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Max retries reached. Please check if Milvus server is running.")
                return False
    return False

def create_collection(vector_dim: int = 1000):
    """Tạo collection trong Milvus với dimension động"""
    # Định nghĩa schema cho collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
    ]
    
    schema = CollectionSchema(fields, description="Movie TF-IDF vectors")
    
    # Tạo collection nếu chưa tồn tại
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
        collection.drop()
        print(f"Dropped existing collection: {COLLECTION_NAME}")
    
    collection = Collection(COLLECTION_NAME, schema)
    print(f"Created collection: {COLLECTION_NAME} with dimension: {vector_dim}")
    
    # Tạo index cho vector field với tham số tối ưu
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": min(128, max(16, vector_dim // 50))}  # Điều chỉnh nlist theo vector_dim
    }
    collection.create_index("vector", index_params)
    print("Created index for vector field")
    
    return collection

def get_collection():
    """Lấy collection hiện có"""
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)
    else:
        raise Exception(f"Collection {COLLECTION_NAME} does not exist")

def dict_to_vector(tfidf_dict: dict, all_words: list) -> list:
    """Chuyển đổi TF-IDF dictionary thành vector với độ dài cố định - tối ưu memory"""
    # Sử dụng numpy để tối ưu memory
    vector = np.zeros(len(all_words), dtype=np.float32)
    
    # Chỉ cập nhật các giá trị non-zero
    for word, value in tfidf_dict.items():
        if word in all_words:
            idx = all_words.index(word)  # Có thể tối ưu bằng dictionary lookup
            vector[idx] = value
    
    return vector.tolist()

def dict_to_vector_optimized(tfidf_dict: dict, word_to_idx: dict) -> list:
    """Phiên bản tối ưu với word index mapping"""
    vector = np.zeros(len(word_to_idx), dtype=np.float32)
    
    for word, value in tfidf_dict.items():
        if word in word_to_idx:
            vector[word_to_idx[word]] = value
    
    return vector.tolist()

def vector_to_dict(vector: list, all_words: list, threshold: float = 1e-8) -> dict:
    """Chuyển đổi vector thành TF-IDF dictionary - chỉ lưu giá trị > threshold"""
    return {word: val for word, val in zip(all_words, vector) if val > threshold}

def create_word_to_idx_mapping(all_words: list) -> dict:
    """Tạo mapping từ word đến index để tối ưu lookup"""
    return {word: idx for idx, word in enumerate(all_words)}