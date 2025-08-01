import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.movie_model import Movie
from typing import List, Optional
from repositories.movie_repo import MovieREPO

movieRepo = MovieREPO()

class MovieService:
    def get_all_movies(self) -> List[Movie]:
        return movieRepo.get_all_movies()

    def get_movie_by_id(self, movie_id: int) -> Optional[Movie]:
        movie = movieRepo.get_movie_by_id(movie_id)
        return movie

    def add_movie(self, movie: Movie) -> None:
        movieRepo.insert_movie(movie.model_dump())
        return {"message": "Movie added"}

    def update_movie(self, movie_id: int, movie: Movie) -> None:
        movieRepo.update_movie(movie_id, movie.model_dump())
        return {"message": "Movie updated"}

    def delete_movie(self, movie_id: int) -> None:
        movieRepo.delete_movie(movie_id)
        return {"message": "Movie deleted"}