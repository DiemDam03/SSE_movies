import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from repositories.concrete.milvus_repo import MilvusREPO
from repositories.concrete.postgres_repo import PostgresREPO
from core.utilities import VectorHandler
from models.search_result_model import SearchResult
import core.tfidf as tfidf

class SearchService:
    def __init__(self, postgres_repo=None, milvus_repo=None, vec_handler=None) -> None:
        self.postgres_repo = postgres_repo if postgres_repo else PostgresREPO()
        self.milvus_repo = milvus_repo if milvus_repo else MilvusREPO()
        self.vec_handler = vec_handler if vec_handler else VectorHandler()

    def search_top_k_movie(self, query: str, top_k: int) -> list[SearchResult]:

        if not query or query.strip():
            return []
        
        corpus = self.postgres_repo.get_corpus()
        if not corpus:
            return {"message": "No movie in database!"}
        
        if not self.milvus_repo.unique_words or not self.milvus_repo.idf_dict:
            self.milvus_repo.refresh_collection_state()

        unique_words = self.milvus_repo.unique_words
        idf_dict = self.milvus_repo.idf_dict

        if not unique_words or not idf_dict:    
            return []

        query_vocab = tfidf.create_vocab_single(query) 
        query_tf = tfidf.compute_tf_single(query_vocab)
        query_tfidf = tfidf.compute_tfidf_single(query_tf, idf_dict)
        query_vector = self.vec_handler.converting_tfidf_to_fixed_dim_vector(query_tfidf, unique_words)

        return self.milvus_repo.search_top_k_movie(query_vector, top_k)