import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from abc import ABC, abstractmethod
from models.search_result_model import SearchResult
from models.movie_model import Movie


class VectorREPO(ABC):
    @abstractmethod
    def connect_to_milvus(self) -> None:
        pass
    @abstractmethod
    def create_collection(self, vector_dim: int) -> None:
        pass
    @abstractmethod
    def search_top_k_movie(self, query: list[float], top_k: int) -> list[SearchResult]:
        pass
    @abstractmethod
    def store_vectors_to_milvus(self) -> None:
        pass
