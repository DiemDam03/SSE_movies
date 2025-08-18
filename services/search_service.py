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
        # Fix the condition - should be "if not query or not query.strip()"
        if not query or not query.strip():
            return []
        
        corpus = self.postgres_repo.get_corpus()
        if not corpus:
            return []
        
        # Ensure the Milvus repository has the latest state
        if not self.milvus_repo.unique_words or not self.milvus_repo.idf_dict:
            self.milvus_repo.refresh_collection_state()

        unique_words = self.milvus_repo.unique_words
        idf_dict = self.milvus_repo.idf_dict

        if not unique_words or not idf_dict:    
            return []

        # Check collection exists and get its info
        collection_info = self.milvus_repo.get_collection_info()
        if not collection_info['exists']:
            print("Collection does not exist. Please initialize the data first.")
            return []

        # Generate query vector
        query_vocab = tfidf.create_vocab_single(query) 
        query_tf = tfidf.compute_tf_single(query_vocab)
        query_tfidf = tfidf.compute_tfidf_single(query_tf, idf_dict)
        query_vector = self.vec_handler.converting_tfidf_to_fixed_dim_vector(query_tfidf, unique_words)

        # Verify dimension compatibility
        expected_dim = collection_info['dimension']
        if len(query_vector) != expected_dim:
            print(f"Dimension mismatch: query vector has {len(query_vector)} dimensions, "
                  f"but collection expects {expected_dim}. Refreshing collection state...")
            
            # Try to refresh and regenerate
            self.milvus_repo.refresh_collection_state()
            unique_words = self.milvus_repo.unique_words
            idf_dict = self.milvus_repo.idf_dict
            
            if unique_words and idf_dict:
                query_tfidf = tfidf.compute_tfidf_single(query_tf, idf_dict)
                query_vector = self.vec_handler.converting_tfidf_to_fixed_dim_vector(query_tfidf, unique_words)
                
                if len(query_vector) != expected_dim:
                    print("Dimension still mismatched after refresh. Collection may need rebuilding.")
                    return []
            else:
                return []

        try:
            return self.milvus_repo.search_top_k_movie(query_vector, top_k)
        except Exception as e:
            print(f"Search failed: {e}")
            return []