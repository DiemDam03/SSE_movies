
from fastapi import FastAPI, HTTPException, Depends
from typing import List
from models.movie import Movie, MovieCreate, MovieUpdate, MovieInDB
from repositories.movie_repository import PostgresMovieRepository
from repositories.vector_repository import MilvusVectorRepository
from services.movie_service import MovieService
from services.initialization_service import InitializationService
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Movie CRUD API")

# Global service instance
_movie_service = None

def get_movie_service():
    global _movie_service
    if _movie_service is None:
        movie_repo = PostgresMovieRepository()
        vector_repo = MilvusVectorRepository()
        _movie_service = MovieService(movie_repo, vector_repo)
    return _movie_service

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup if needed"""
    logger = logging.getLogger(__name__)
    try:
        movie_repo = PostgresMovieRepository()
        vector_repo = MilvusVectorRepository()
        
        # Check if initialization is needed
        existing_movies = movie_repo.get_all()
        if not existing_movies:
            logger.info("No movies found, running initialization...")
            init_service = InitializationService(movie_repo, vector_repo)
            init_service.initialize_database()
        else:
            logger.info(f"Found {len(existing_movies)} existing movies, skipping initialization")
            
    except Exception as e:
        logger.error(f"Startup initialization failed: {e}")
        # Don't fail startup, but log the error

@app.get("/movies/", response_model=List[MovieInDB])
def get_all_movies(service: MovieService = Depends(get_movie_service)):
    return service.get_all_movies()

@app.get("/movies/{movie_id}", response_model=MovieInDB)
def get_movie_by_id(movie_id: int, service: MovieService = Depends(get_movie_service)):
    movie = service.get_movie_by_id(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie

@app.post("/movies/", response_model=MovieInDB)
def add_movie(movie: MovieCreate, service: MovieService = Depends(get_movie_service)):
    return service.create_movie(movie)

@app.put("/movies/{movie_id}", response_model=MovieInDB)
def update_movie(movie_id: int, movie: MovieUpdate, service: MovieService = Depends(get_movie_service)):
    result = service.update_movie(movie_id, movie)
    if result is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return result

@app.delete("/movies/{movie_id}")
def delete_movie(movie_id: int, service: MovieService = Depends(get_movie_service)):
    if not service.delete_movie(movie_id):
        raise HTTPException(status_code=404, detail="Movie not found")
    return {"message": "Movie deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_crud:app", host="0.0.0.0", port=8001, reload=True)
