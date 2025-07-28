from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import db
import tfidf
from db import movie_dataset_processing_from_sqlite, generate_vectors
import uvicorn

app = FastAPI(title="Movie Search")

class SearchResult(BaseModel):
    index: int
    score: float
    text: str

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

import milvus_db  # thay vì import db
...

@app.get("/search", response_model=List[SearchResult])
def search_movies(q: str = Query(...), top_k: int = 5):
    corpus_texts = db.movie_dataset_processing_from_sqlite()
    idf_dict = tfidf.compute_idf_single(corpus_texts)
    query_vocab = tfidf.create_vocab_single(q)
    query_tf = tfidf.compute_tf_single(query_vocab)
    query_tfidf = tfidf.compute_tfidf_single(query_tf, idf_dict)

    top_scores = milvus_db.search(query_tfidf, top_k)

    results = []
    for idx, score, text in top_scores:
        results.append(SearchResult(index=idx, score=score, text=text))
    return results


if __name__ == "__main__":
    texts, vectors = generate_vectors()
    milvus_db.create_collection()
    milvus_db.insert_data(texts, vectors)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

