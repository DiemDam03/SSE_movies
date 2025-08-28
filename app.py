import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI, Query, HTTPException
from services.movie_service import MovieService
from models.movie_model import Movie

from repositories.concrete.milvus_repo import MilvusREPO
from repositories.concrete.postgres_repo import PostgresREPO
from core.utilities import VectorHandler
from models.search_result_model import SearchResult
from services.search_service import SearchService

app = FastAPI(title="APP")

pg_repo = PostgresREPO()
mv_repo = MilvusREPO()
vec_handler = VectorHandler()
movie_service = MovieService(pg_repo, mv_repo)
search_service = SearchService(pg_repo, mv_repo, vec_handler)

@app.get("/search", response_model=list[SearchResult])
def search_movies(query: str = Query(...), top_k: int = 5):
    return search_service.search_top_k_movie(query, top_k)

@app.get("/movies/", response_model=list[Movie])
def get_all_movies():
    return movie_service.get_all_movies()

@app.get("/movies/{movie_id}", response_model=Movie)
def get_movie_by_id(movie_id: int):
    movie = movie_service.get_movie_by_id(movie_id)
    if movie is None:   
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie

@app.post("/movies/")
def add_movie(movie: Movie):
    movie_service.add_movie(movie.model_dump())
    return {"message": "Movie added"}

@app.put("/movies/{movie_id}")
def update_movie(movie_id: int, movie: Movie):
    movie_service.update_movie(movie_id, movie.model_dump())
    return {"message": "Movie updated"}

@app.delete("/movies/{movie_id}")
def delete_movie(movie_id: int):
    movie_service.delete_movie(movie_id)
    return {"message": "Movie deleted"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)