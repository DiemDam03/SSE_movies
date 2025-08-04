import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from repositories.concrete.milvus_repo import MilvusREPO
from repositories.concrete.postgres_repo import PostgresREPO
from core.utilities import VectorHandler
from models.search_result_model import SearchResult
import core.tfidf as tfidf

class SearchService:
    def __init__(self, postgres=PostgresREPO, milvus=MilvusREPO, handler=VectorHandler) -> None:
        self.postgres_repo = PostgresREPO()
        self.milvus_repo = MilvusREPO()
        self.vec_handler = VectorHandler()
        self.idf_dict = None
        self.unique_words = None

    def search_top_k_movie(self, query: str, top_k: int) -> list[SearchResult]:
        if self.unique_words is None or self.idf_dict is None:
            corpus = self.postgres_repo.get_corpus()
            _, self.idf_dict = self.vec_handler.generate_tfidf(corpus)
            self.unique_words = self.vec_handler.get_unique_words(corpus)

        query_vocab = tfidf.create_vocab_single(query)
        query_tf = tfidf.compute_tf_single(query_vocab)
        query_tfidf = tfidf.compute_tfidf_single(query_tf, self.idf_dict)
        query_vector = self.vec_handler.converting_tfidf_to_fixed_dim_vector(query_tfidf, self.unique_words)

        return self.milvus_repo.search_top_k_movie(query_vector, top_k)