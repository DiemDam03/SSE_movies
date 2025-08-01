import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from repositories.concrete.milvus_repo import MilvusREPO
from repositories.concrete.postgres_repo import PostgresREPO
from core.utilities import VectorHandler
from models.search_result_model import SearchResult
import core.tfidf as tfidf

from pymilvus import Collection


class SearchService:
    def __init__(self, postgres = PostgresREPO, milvus = MilvusREPO) -> None:
        self.postgres_repo = postgres
        self.milvus_repo = milvus
        self.idf_dict = None
        self.unique_words = None

    def search_top_k_movie(self, query: list[float], top_k: int) -> list[SearchResult]:
        handler = VectorHandler(self.postgres_repo)             #Vectorhandler làm gì có () mà truyền vào
        if self.unique_words is None or self.idf_dict is None:
            corpus = self.postgres_repo.loading_data_from_postgres()
            _, self.idf_dict = handler.generate_tfidf(corpus) # cái này ko lưu vào postgres à
            self.unique_words = handler.get_unique_words(corpus)

        query_vocab = tfidf.create_vocab_single(query)
        query_tf = tfidf.compute_tf_single(query_vocab)
        query_tfidf = tfidf.compute_tfidf_single(query_tf, self.idf_dict)
        query_vector = handler.converting_tfidf_to_fixed_dim_vector(query_tfidf, self.unique_words)

        self.milvus_repo.search_top_k_movie(query_vector, top_k)
    
    # def filter_by_genre(self, genre: str):
    #     return self.filter_by_genre(self, genre)