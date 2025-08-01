
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from db import dbm
from pymilvus import Collection
import dependencies.tfidf as tfidf

app = FastAPI(title="Movie Search")

class SearchResult(BaseModel):
    index: int
    score: float
    text: str

@app.get("/search", response_model=List[SearchResult])
def search_movies(q: str = Query(...), top_k: int = 5):
    if dbm.unique_words is None or dbm.idf_dict is None:
        dbm.update_dict_after_crud()

    query_vocab = tfidf.create_vocab_single(q)
    query_tf = tfidf.compute_tf_single(query_vocab)
    query_tfidf = tfidf.compute_tfidf_single(query_tf, dbm.idf_dict)
    query_vector = dbm.converting_tfidf_to_fixed_dim_vector(query_tfidf, dbm.unique_words)

    dbm.connect_to_milvus()
    collection = Collection("movie_collection")
    collection.load()

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["id", "movieId", "text"]
    )

    top_hits = results[0]
    response = [
        SearchResult(index=hit.id, score=hit.distance, text=hit.entity.get("text"))
        for hit in top_hits
    ]

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_search:app", host="0.0.0.0", port=8000, reload=True)