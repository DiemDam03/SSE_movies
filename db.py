import os
import pandas as pd
import tfidf
import sqlite3
import json
from pymilvus import Collection, utility
import milvus_config
import gc  # Garbage collection
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DATASET_PATH = os.path.join(BASE_DIR, "movies.csv")
SQL_DB_PATH = os.path.join(BASE_DIR, "movies.sqlite")
VOCAB_PATH = os.path.join(BASE_DIR, "vocabulary.json")

def convert_csv_to_sqlite(csv_path, db_path):
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    df.to_sql("movies", conn, if_exists="replace", index=False)
    conn.close()

def movie_dataset_processing_from_sqlite():
    conn = sqlite3.connect(SQL_DB_PATH)
    df = pd.read_sql_query("SELECT title, genres FROM movies", conn)
    df['text'] = df['title'] + ' | ' + df['genres']
    conn.close()
    return df['text'].tolist()

def save_vocabulary(all_words: list):
    """Lưu vocabulary vào file JSON"""
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_words, f, ensure_ascii=False, indent=2)

def load_vocabulary() -> list:
    """Tải vocabulary từ file JSON"""
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_word_mapping(word_to_idx: dict):
    """Lưu word-to-index mapping vào file JSON"""
    mapping_path = os.path.join(BASE_DIR, "word_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(word_to_idx, f, ensure_ascii=False, indent=2)

def load_word_mapping() -> dict:
    """Tải word-to-index mapping từ file JSON"""
    mapping_path = os.path.join(BASE_DIR, "word_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def generate_vectors_and_vocabulary():
    """Tạo vectors và vocabulary từ dataset"""
    dataset = movie_dataset_processing_from_sqlite()
    vocab_all = tfidf.create_vocab_all(dataset)
    tf_all = tfidf.compute_tf_all(vocab_all)
    idf_dict = tfidf.compute_idf_single(dataset)
    tfidf_all = tfidf.compute_tfidf_all(tf_all, idf_dict)
    
    # Tạo vocabulary tổng hợp từ tất cả documents
    all_words = list(idf_dict.keys())
    all_words.sort()  # Sắp xếp để đảm bảo thứ tự nhất quán
    
    return dataset, tfidf_all, all_words

def storing_vectors_milvus(dataset: list[str], vectorized_dataset: list[dict], all_words: list):
    """Lưu trữ vectors vào Milvus với batch processing và connection resilience"""
    if not milvus_config.connect_milvus():
        raise Exception("Cannot connect to Milvus")
    
    # Tạo collection với dimension đúng
    vector_dim = len(all_words)
    collection = milvus_config.create_collection(vector_dim)
    
    # Tạo word-to-index mapping để tối ưu lookup
    word_to_idx = milvus_config.create_word_to_idx_mapping(all_words)
    
    # Insert data theo batch rất nhỏ và với retry logic
    batch_size = 20  # Giảm batch size xuống 20
    total_records = len(dataset)
    retry_count = 0
    max_retries = 3
    
    i = 0
    while i < total_records:
        end_idx = min(i + batch_size, total_records)
        
        try:
            # Xử lý batch hiện tại
            batch_texts = []
            batch_vectors = []
            
            for j in range(i, end_idx):
                text = dataset[j]
                tfidf_dict = vectorized_dataset[j]
                
                # Sử dụng optimized conversion
                vector = milvus_config.dict_to_vector_optimized(tfidf_dict, word_to_idx)
                
                batch_texts.append(text)
                batch_vectors.append(vector)
            
            # Insert batch với retry logic
            success = False
            for retry in range(max_retries):
                try:
                    # Reconnect if needed
                    if retry > 0:
                        print(f"Reconnecting to Milvus... (retry {retry})")
                        if not milvus_config.connect_milvus():
                            raise Exception("Cannot reconnect to Milvus")
                        collection = milvus_config.get_collection()
                    
                    data = [batch_texts, batch_vectors]
                    collection.insert(data)
                    print(f"Inserted batch {i//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}: {len(batch_texts)} records")
                    success = True
                    retry_count = 0  # Reset retry count on success
                    break
                    
                except Exception as batch_error:
                    print(f"Error inserting batch {i//batch_size + 1} (retry {retry + 1}): {batch_error}")
                    if retry < max_retries - 1:
                        time.sleep(5)  # Wait before retry
                    else:
                        print(f"Failed to insert batch after {max_retries} retries")
                        
            if not success:
                retry_count += 1
                if retry_count > 5:  # Too many consecutive failures
                    print("Too many consecutive failures. Stopping.")
                    break
                print("Continuing with next batch...")
                
            # Cleanup memory
            del batch_texts, batch_vectors
            gc.collect()
            
            # Move to next batch only if successful
            if success:
                i = end_idx
                
                # Flush occasionally
                if (i // batch_size) % 10 == 0:
                    try:
                        collection.flush()
                        print("Flushed data to Milvus")
                    except Exception as flush_error:
                        print(f"Flush error (continuing): {flush_error}")
            
        except Exception as e:
            print(f"General error processing batch {i//batch_size + 1}: {e}")
            retry_count += 1
            if retry_count > 5:
                print("Too many errors. Stopping.")
                break
            time.sleep(5)
    
    # Final flush
    try:
        collection.flush()
        print("Final flush completed")
    except Exception as e:
        print(f"Final flush error: {e}")
    
    # Lưu vocabulary và word mapping
    save_vocabulary(all_words)
    save_word_mapping(word_to_idx)
    
    print(f"Process completed. Attempted to insert {total_records} vectors into Milvus collection")

def loading_vectors_milvus() -> tuple[list[dict], list[str]]:
    """Tải vectors từ Milvus"""
    if not milvus_config.connect_milvus():
        raise Exception("Cannot connect to Milvus")
    
    if not utility.has_collection(milvus_config.COLLECTION_NAME):
        raise FileNotFoundError("Milvus collection not found. Please initialize it first.")
    
    collection = milvus_config.get_collection()
    collection.load()
    
    # Load vocabulary
    all_words = load_vocabulary()
    if not all_words:
        raise FileNotFoundError("Vocabulary file not found. Please initialize database first.")
    
    # Query tất cả data từ collection
    results = collection.query(
        expr="id >= 0",
        output_fields=["text", "vector"]
    )
    
    vector_store = []
    for result in results:
        text = result["text"]
        vector = result["vector"]
        tfidf_dict = milvus_config.vector_to_dict(vector, all_words)
        vector_store.append({
            "text": text,
            "vector": tfidf_dict
        })
    
    return vector_store, all_words

def search_vectors_milvus(query_vector: list, top_k: int = 5) -> list[tuple[int, float, str]]:
    """Tìm kiếm vectors tương tự trong Milvus"""
    if not milvus_config.connect_milvus():
        raise Exception("Cannot connect to Milvus")
    
    collection = milvus_config.get_collection()
    collection.load()
    
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }
    
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )
    
    search_results = []
    for hits in results:
        for hit in hits:
            search_results.append((hit.id, hit.score, hit.entity.get("text")))
    
    return search_results

def initialize_database():
    """Khởi tạo database với Milvus"""
    try:
        if not milvus_config.connect_milvus():
            print("Cannot connect to Milvus. Please make sure Milvus server is running.")
            return
        
        # Kiểm tra xem collection đã tồn tại chưa
        if utility.has_collection(milvus_config.COLLECTION_NAME):
            collection = milvus_config.get_collection()
            collection.load()
            count = collection.num_entities
            if count > 0:
                print(f"Milvus collection already exists with {count} vectors.")
                return
        
        print("Creating Milvus vector database...")
        dataset, vectorized_dataset, all_words = generate_vectors_and_vocabulary()
        print(f"Generated vectors for {len(dataset)} documents with vocabulary size: {len(all_words)}")
        
        storing_vectors_milvus(dataset, vectorized_dataset, all_words)
        print("Milvus vector database created successfully!")
        
    except Exception as e:
        print(f"Error initializing Milvus database: {e}")
        import traceback
        traceback.print_exc()

def get_all_movies():
    conn = sqlite3.connect(SQL_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT movieId, title, genres FROM movies")
    movies = [{"id": row[0], "title": row[1], "genres": row[2]} for row in c.fetchall()]
    conn.close()
    return movies

def get_movie_by_id(movie_id: int):
    conn = sqlite3.connect(SQL_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT movieId, title, genres FROM movies WHERE movieId = ?", (movie_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return {"id": row[0], "title": row[1], "genres": row[2]}
    return None

def insert_movie(movie: dict):
    conn = sqlite3.connect(SQL_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO movies (movieId, title, genres) VALUES (?, ?, ?)", 
              (movie["id"], movie["title"], movie["genres"]))
    conn.commit()
    conn.close()

def update_movie(movie_id: int, movie: dict):
    conn = sqlite3.connect(SQL_DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE movies SET title = ?, genres = ? WHERE movieId = ?", 
              (movie["title"], movie["genres"], movie_id))
    conn.commit()
    conn.close()

def delete_movie(movie_id: int):
    conn = sqlite3.connect(SQL_DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM movies WHERE movieId = ?", (movie_id,))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    convert_csv_to_sqlite(CSV_DATASET_PATH, SQL_DB_PATH)
    initialize_database()