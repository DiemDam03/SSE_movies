from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from database.db import DatabaseManager
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

dbm = DatabaseManager()
app = FastAPI(title="CRUD")

class Movie(BaseModel):
    id: int
    title: str
    genres: str

@app.get("/movies/", response_model=List[Movie])
def get_all_movies():
    return dbm.get_all_movies()

@app.get("/movies/{movie_id}", response_model=Movie)
def get_movie_by_id(movie_id: int):
    movie = dbm.get_movie_by_id(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie

@app.post("/movies/")
def add_movie(movie: Movie):
    dbm.insert_movie(movie.model_dump())
    return {"message": "Movie added"}

@app.put("/movies/{movie_id}")
def update_movie(movie_id: int, movie: Movie):
    dbm.update_movie(movie_id, movie.model_dump())
    return {"message": "Movie updated"}

@app.delete("/movies/{movie_id}")
def delete_movie(movie_id: int):
    dbm.delete_movie(movie_id)
    return {"message": "Movie deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_crud:app", host="0.0.0.0", port=8001, reload=True)