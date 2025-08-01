import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from repositories.interfaces.vector_repo import VectorREPO
from typing import List, Optional
from models.movie_model import Movie
from models.search_result_model import SearchResult
from pymilvus import Collection
from dependencies.utilities import utilities
import dependencies.tfidf as tfidf
from repositories.concrete.milvus_repo import MilvusREPO

mv = MilvusREPO()
uti = utilities()

class ServiceREPO(VectorREPO):
    def search_top_k_movie(self, query: List[float], top_k: int) -> List[Movie]:
        if dbm.unique_words is None or dbm.idf_dict is None:
            uti.update_dict_after_crud()

        query_vocab = tfidf.create_vocab_single(q)
        query_tf = tfidf.compute_tf_single(query_vocab)
        query_tfidf = tfidf.compute_tfidf_single(query_tf, dbm.idf_dict)
        query_vector = uti.converting_tfidf_to_fixed_dim_vector(query_tfidf, dbm.unique_words)

        mv.connect_to_milvus()
        collection = Collection("movie_collection")
        collection.load()

        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["id", "movieId", "text"]
        )

        top_hits = results[0]
        response = [
            SearchResult(index=hit.id, score=hit.distance, text=hit.entity.get("text"))
            for hit in top_hits
        ]

        return response
        pass