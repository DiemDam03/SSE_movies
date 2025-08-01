import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from abc import ABC, abstractmethod
from models.movie_model import Movie
from typing import List, Optional

class VectorREPO(ABC):
    @abstractmethod
    def search_top_k_movie(self, query: List[float], top_k: int) -> List[Movie]:
        pass
    @abstractmethod
    def store_vectors_to_milvus(self) -> None:
        pass
    @abstractmethod
    def load_vectors_from_milvus(self) -> List[float]:
        pass
    # @abstractmethod
    # def add_vector(self, movie_id: int,  movie: Movie) -> None:
    #     pass
    # @abstractmethod
    # def update_vector(self, movie_id: int, movie: Movie) -> None:
    #     pass
    # @abstractmethod
    # def delete_vector(self, movie_id: int) -> None:
    #     pass