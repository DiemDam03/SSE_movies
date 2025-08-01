import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.movie_model import Movie
from repositories.concrete.milvus_repo import MilvusREPO
from repositories.concrete.postgres_repo import PostgresREPO
from core.utilities import VectorHandler

class MovieService:
    def __init__(self, postgres = PostgresREPO, milvus = MilvusREPO) -> None:
        self.postgres_repo = postgres
        self.milvus_repo = milvus

    def get_all_movies(self) -> list[Movie]:
        return self.postgres_repo.get_all_movies()

    def get_movie_by_id(self, movie_id: int) -> Movie | None:
        movie = self.postgres_repo.get_movie_by_id(movie_id)
        return movie

    def add_movie(self, movie: dict) -> None:
        self.postgres_repo.add_movie(movie)
        self.sync_with_milvus()
        return {"message": "Movie added"}

    def update_movie(self, movie_id: int, movie: dict) -> None:
        self.postgres_repo.update_movie(movie_id, movie)
        self.sync_with_milvus()
        return {"message": "Movie updated"}

    def delete_movie(self, movie_id: int) -> None:
        self.postgres_repo.delete_movie(movie_id)
        self.sync_with_milvus()
        return {"message": "Movie deleted"}

    def sync_with_milvus(self) -> None:
        handler = VectorHandler(self.postgres_repo) #Vectorhandler làm gì có () mà truyền vào
        
        corpus = self.postgres_repo.loading_data_from_postgres(self)
        vectors, unique_words = handler.generate_vectors(corpus)

        self.milvus_repo.create_collection(len(unique_words))

        # có gì khác load data bằng postgres_repo mà phải làm lại
        movie_data = []
        conn = self.postgres_repo.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT movieId, title, genres FROM movies ORDER BY movieId")
        movie_data = cursor.fetchall() #? tạo movie data ở trên làm gì
        cursor.close()
        conn.close()

        self.milvus_repo.store_vectors_to_milvus(vectors, movie_data)

