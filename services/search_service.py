import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from repositories.interfaces.vector_repo import VectorREPO
from typing import List, Optional
from models.movie_model import Movie


class MovieService(VectorREPO):
    def search_top_k_movie(self, query: List[float], top_k: int) -> List[Movie]:
        # chưa implelment
        pass