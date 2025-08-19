import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.movie_model import Movie
from repositories.concrete.milvus_repo import MilvusREPO
from repositories.concrete.postgres_repo import PostgresREPO
from services.data_service import DataManager

class MovieService:
    def __init__(self, postgres_repo=None, milvus_repo=None) -> None:
        self.postgres_repo = postgres_repo if postgres_repo else PostgresREPO()
        self.milvus_repo = milvus_repo if milvus_repo else MilvusREPO()
        # self.milvus_repo.load state?
        self.milvus_repo.load_state()

    def get_all_movies(self) -> list[Movie]:
        return self.postgres_repo.get_all_movies()

    def get_movie_by_id(self, movie_id: int) -> Movie | None:
        movie = self.postgres_repo.get_movie_by_id(movie_id)
        return movie

    def add_movie(self, movie: dict) -> dict:
        self.postgres_repo.add_movie(movie)
        self.milvus_repo.add_movie(movie) 

    def update_movie(self, movie_id: int, movie: dict) -> dict:
        self.postgres_repo.update_movie(movie_id, movie)
        updated_movie_data = {
            'id': movie_id,
            'title': movie['title'],
            'genres': movie['genres']
        }
        self.milvus_repo.update_movie(movie_id, updated_movie_data)

    def delete_movie(self, movie_id: int) -> dict:
        existing_movie = self.postgres_repo.get_movie_by_id(movie_id)
        if not existing_movie:
            raise Exception(f"Movie with ID {movie_id} not found")
        self.milvus_repo.delete_movie(movie_id)
        self.postgres_repo.delete_movie(movie_id)
        corpus = self.postgres_repo.get_corpus()
        if corpus:  
            self.milvus_repo.refresh_collection_state()
