from typing import List, Dict, Any
from repositories.vector_repository import VectorRepositoryInterface
from repositories.movie_repository import MovieRepositoryInterface
from core.tfidf import *
import logging

logger = logging.getLogger(__name__)

class SearchResult:
    def __init__(self, index: int, score: float, text: str, movie_id: int):
        self.index = index
        self.score = score
        self.text = text
        self.movie_id = movie_id

class SearchService:
    def __init__(self, movie_repo: MovieRepositoryInterface, vector_repo: VectorRepositoryInterface):
        self.movie_repo = movie_repo
        self.vector_repo = vector_repo
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        try:
            # Get corpus and compute IDF
            corpus = self.movie_repo.get_corpus()
            if not corpus:
                return []
            
            idf_dict = compute_idf_single(corpus)
            
            # Process query
            query_vocab = create_vocab_single(query)
            query_tf = compute_tf_single(query_vocab)
            query_tfidf = compute_tfidf_single(query_tf, idf_dict)
            
            # Get vectors from Milvus
            vector_store = self.vector_repo.get_all_vectors()
            
            # Convert stored vectors back to TF-IDF format for ranking
            tfidf_list = []
            for item in vector_store:
                vector = item["vector"]
                # This is a simplified conversion - in practice you'd need to maintain word mappings
                tfidf_dict = {f"word_{i}": val for i, val in enumerate(vector) if val > 0}
                tfidf_list.append(tfidf_dict)
            
            # Rank results
            top_scores = ranking(query_tfidf, tfidf_list, top_k)
            
            results = []
            for idx, score in top_scores:
                if idx < len(vector_store):
                    item = vector_store[idx]
                    results.append(SearchResult(
                        index=idx,
                        score=score,
                        text=item["text"],
                        movie_id=item["movieId"]
                    ))
            
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise