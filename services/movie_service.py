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
        # self.milvus_repo.refresh_collection_state() # đã có trong add
        self.milvus_repo.add_movie(movie) # này sai 
        
    # cái này cũng sai
    # hẳn là hàm milvus update và add
    # hay là chỉ sai add, vì update cũng mò vào colleciton mà trc đó đã ko có do ko bỏ vào lúc add => ko, nó sai do bản thân sai.
    # (code=65535, message=the length(10462) of float data should divide the dim(10458))> 
    # update sai do sử dụng lại func add mà ko tự insert, upsert.-> bỏ add, tự làm xem còn lỗi ko .

    # vẫn là sai dimension, cần update dimension.
    # tại sao delete ko cần quan tâm dimesion? tại nó ko vào colleciton xem, nó lấy id xong xóa luôn cái ô đó
    # làm sao để update dimension? create lại collection lại từ đầu? tốn nhiều chi phí => cách này ko ổn.
    # 
    # update schema để thay đổi vector dim? update schema có thay đổi vec dim của collection? 
    def update_movie(self, movie_id: int, movie: dict) -> dict:
        existing_movie = self.postgres_repo.get_movie_by_id(movie_id)
        if not existing_movie:
            raise Exception(f"Movie with ID {movie_id} not found")
        self.postgres_repo.update_movie(movie_id, movie)
        corpus = self.postgres_repo.get_corpus()
        self.milvus_repo.refresh_collection_state(corpus)
        updated_movie_data = {
            'id': movie_id,
            'title': movie['title'],
            'genres': movie['genres']
        }
        self.milvus_repo.update_movie(movie_id, updated_movie_data, corpus)

    def delete_movie(self, movie_id: int) -> dict:
        existing_movie = self.postgres_repo.get_movie_by_id(movie_id)
        if not existing_movie:
            raise Exception(f"Movie with ID {movie_id} not found")
        self.milvus_repo.delete_movie(movie_id)
        self.postgres_repo.delete_movie(movie_id)
        corpus = self.postgres_repo.get_corpus()
        if corpus:  
            self.milvus_repo.refresh_collection_state(corpus)
