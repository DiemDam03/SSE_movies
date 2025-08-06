import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.movie_model import Movie
from repositories.concrete.milvus_repo import MilvusREPO
from repositories.concrete.postgres_repo import PostgresREPO
from services.data_service import DataManager

class MovieService:
    def __init__(self, postgres_repo=None, milvus_repo=None, data_manager=None) -> None:
        self.postgres_repo = postgres_repo if postgres_repo else PostgresREPO()
        self.milvus_repo = milvus_repo if milvus_repo else MilvusREPO()
        self.data_manager = data_manager if data_manager else DataManager()

    def get_all_movies(self) -> list[Movie]:
        return self.postgres_repo.get_all_movies()

    def get_movie_by_id(self, movie_id: int) -> Movie | None:
        movie = self.postgres_repo.get_movie_by_id(movie_id)
        return movie

    def add_movie(self, movie: dict) -> dict:
        self.postgres_repo.add_movie(movie)
        self.data_manager.sync_postgres_milvus()
        return {"message": "Movie added"}

    def update_movie(self, movie_id: int, movie: dict) -> dict:
        self.postgres_repo.update_movie(movie_id, movie)
        self.data_manager.sync_postgres_milvus()
        return {"message": "Movie updated"}

    def delete_movie(self, movie_id: int) -> dict:
        self.postgres_repo.delete_movie(movie_id)
        self.data_manager.sync_postgres_milvus()
        return {"message": "Movie deleted"}

    