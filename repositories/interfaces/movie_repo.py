import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from abc import ABC, abstractmethod
from models.movie_model import Movie


class MovieREPO(ABC):
    @abstractmethod
    def connect_to_postgres(self) -> None:
        pass
    @abstractmethod
    def create_table(self) -> None:
        pass
    @abstractmethod
    def get_all_movies(self) -> list[Movie]:
        pass
    @abstractmethod
    def get_movie_by_id(self, movie_id: int) -> Movie | None:
        pass
    @abstractmethod
    def add_movie(self, movie: Movie) -> None:
        pass
    @abstractmethod
    def update_movie(self, movie_id: int, movie: Movie) -> None:
        pass
    @abstractmethod
    def delete_movie(self, movie_id: int) -> None:
        pass