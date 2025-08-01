import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI, Query

from repositories.concrete.milvus_repo import MilvusREPO
from repositories.concrete.postgres_repo import PostgresREPO
from models.search_result_model import SearchResult
from services.search_service import SearchService

app = FastAPI(title="Movie Search")

pg_repo = PostgresREPO()
mv_repo = MilvusREPO()

search_service = SearchService(pg_repo, mv_repo)

@app.get("/search", response_model=list[SearchResult])
def search_movies(query: str = Query(...), top_k: int = 5):
    return search_service.search_top_k_movie(query, top_k)

    pass
if __name__ == "__main__":
    uvicorn.run("user_web:app", host="0.0.0.0", port=8000, reload=True)