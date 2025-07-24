from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import db
import uvicorn



app = FastAPI(title="CRUD")

class Movie(BaseModel):
    id: int
    title: str
    genres: str

@app.get("/movies/", response_model=List[Movie])
def get_all_movies():
    return db.get_all_movies()

@app.get("/movies/{movie_id}", response_model=Movie)
def get_movie_by_id(movie_id: int):
    movie = db.get_movie_by_id(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie

@app.post("/movies/")
def add_movie(movie: Movie):
    db.insert_movie(movie.model_dump())
    db.initialize_database()  
    return {"message": "Movie added"}

@app.put("/movies/{movie_id}")
def update_movie(movie_id: int, movie: Movie):
    db.update_movie(movie_id, movie.model_dump())
    db.initialize_database()
    return {"message": "Movie updated"}

@app.delete("/movies/{movie_id}")
def delete_movie(movie_id: int):
    db.delete_movie(movie_id)
    db.initialize_database()
    return {"message": "Movie deleted"}

if __name__ == "__main__":
    uvicorn.run("db_manager:app", host="0.0.0.0", port=8000, reload=True)