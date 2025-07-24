# from fastapi import FastAPI, Query
# from pydantic import BaseModel
# from typing import List
# import db
# import tfidf

# app = FastAPI(title="Movie Search")

# class SearchResult(BaseModel):
#     index: int
#     score: float
#     text: str

# @app.get("/search", response_model=List[SearchResult])
# def search_movies(q: str = Query(...), top_k: int = 5):
#     vector_store = db.loading_vectors()
#     corpus_texts = [item["text"] for item in vector_store]
#     idf_dict = tfidf.compute_idf_single(corpus_texts)
    
#     query_vocab = tfidf.create_vocab_single(q)
#     query_tf = tfidf.compute_tf_single(query_vocab)
#     query_tfidf = tfidf.compute_tfidf_single(query_tf, idf_dict)
    
#     tfidf_list = [item["vector"] for item in vector_store]
#     top_scores = tfidf.ranking(query_tfidf, tfidf_list, top_k)

#     results = []
#     for idx, score in top_scores:
#         results.append(SearchResult(index=idx, score=score, text=vector_store[idx]["text"]))
#     return results


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List
import db
import tfidf
import milvus_config

app = FastAPI(title="Movie Search with Milvus")

class SearchResult(BaseModel):
    index: int
    score: float
    text: str

@app.get("/search", response_model=List[SearchResult])
def search_movies(q: str = Query(...), top_k: int = 5):
    try:
        # Load vocabulary
        all_words = db.load_vocabulary()
        if not all_words:
            raise HTTPException(status_code=503, detail="Database not initialized. Please run initialization first.")
        
        # Tạo TF-IDF vector cho query
        query_vocab = tfidf.create_vocab_single(q)
        query_tf = tfidf.compute_tf_single(query_vocab)
        
        # Load corpus để tính IDF (có thể cache lại để tối ưu)
        vector_store, _ = db.loading_vectors_milvus()
        corpus_texts = [item["text"] for item in vector_store]
        idf_dict = tfidf.compute_idf_single(corpus_texts)
        
        query_tfidf = tfidf.compute_tfidf_single(query_tf, idf_dict)
        query_vector = milvus_config.dict_to_vector(query_tfidf, all_words)
        
        # Tìm kiếm trong Milvus
        search_results = db.search_vectors_milvus(query_vector, top_k)
        
        results = []
        for idx, score, text in search_results:
            results.append(SearchResult(index=int(idx), score=float(score), text=text))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/search/traditional", response_model=List[SearchResult])
def search_movies_traditional(q: str = Query(...), top_k: int = 5):
    """Phương pháp tìm kiếm truyền thống (backup) sử dụng cosine similarity"""
    try:
        vector_store, _ = db.loading_vectors_milvus()
        corpus_texts = [item["text"] for item in vector_store]
        idf_dict = tfidf.compute_idf_single(corpus_texts)
        
        query_vocab = tfidf.create_vocab_single(q)
        query_tf = tfidf.compute_tf_single(query_vocab)
        query_tfidf = tfidf.compute_tfidf_single(query_tf, idf_dict)
        
        tfidf_list = [item["vector"] for item in vector_store]
        top_scores = tfidf.ranking(query_tfidf, tfidf_list, top_k)

        results = []
        for idx, score in top_scores:
            results.append(SearchResult(index=idx, score=score, text=vector_store[idx]["text"]))
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/status")
def get_status():
    """Kiểm tra trạng thái của hệ thống"""
    try:
        milvus_connected = milvus_config.connect_milvus()
        vocab_exists = len(db.load_vocabulary()) > 0
        
        collection_exists = False
        vector_count = 0
        if milvus_connected:
            from pymilvus import utility
            collection_exists = utility.has_collection(milvus_config.COLLECTION_NAME)
            if collection_exists:
                collection = milvus_config.get_collection()
                collection.load()
                vector_count = collection.num_entities
        
        return {
            "milvus_connected": milvus_connected,
            "collection_exists": collection_exists,
            "vocabulary_loaded": vocab_exists,
            "vector_count": vector_count,
            "status": "ready" if all([milvus_connected, collection_exists, vocab_exists]) else "not_ready"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.post("/initialize")
def initialize_system():
    """Khởi tạo hệ thống vector database"""
    try:
        db.initialize_database()
        return {"message": "System initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)