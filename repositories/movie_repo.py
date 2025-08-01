import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from abc import ABC, abstractmethod
from models.movie_model import Movie
from typing import List, Optional

class MovieREPO(ABC):
    @abstractmethod
    def get_all_movies(self) -> List[Movie]:
        pass
    @abstractmethod
    def get_movie_by_id(self, movie_id: int) -> Optional[Movie]:
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
