from Legacy.db import DatabaseManager
from models.interfaces.icrud import CRUD
from models.crud_model import Movie
from typing import List, Optional

dbm = DatabaseManager()

class CRUDService:
    def get_all_movies(self) -> List[Movie]:
        return dbm.get_all_movies()

    def get_movie_by_id(self, movie_id: int) -> Optional[Movie]:
        movie = dbm.get_movie_by_id(movie_id)
        return movie

    def add_movie(self, movie: Movie) -> None:
        dbm.insert_movie(movie.model_dump())
        return {"message": "Movie added"}

    def update_movie(self, movie_id: int, movie: Movie) -> None:
        dbm.update_movie(movie_id, movie.model_dump())
        return {"message": "Movie updated"}

    def delete_movie(self, movie_id: int) -> None:
        dbm.delete_movie(movie_id)
        return {"message": "Movie deleted"}