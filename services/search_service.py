import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from repositories.interfaces.vector_repo import VectorREPO
from typing import List, Optional
from models.movie_model import Movie


class SearchService(VectorREPO):
    def search_top_k_movie(self, query: List[float], top_k: int) -> List[Movie]:
        # chưa implelment
        return self.search_top_k_movie(self, query, top_k)
    
    # def filter_by_genre(self, genre: str):
    #     return self.filter_by_genre(self, genre)