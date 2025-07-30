from typing import List, Optional
from models.movie import Movie, MovieCreate, MovieUpdate, MovieInDB
from repositories.movie_repository import MovieRepositoryInterface
from repositories.vector_repository import VectorRepositoryInterface
from core.tfidf import *
import logging

logger = logging.getLogger(__name__)

class MovieService:
    def __init__(self, movie_repo: MovieRepositoryInterface, vector_repo: VectorRepositoryInterface):
        self.movie_repo = movie_repo
        self.vector_repo = vector_repo
        self._idf_dict = None
        self._unique_words = None
    
    def get_all_movies(self) -> List[MovieInDB]:
        return self.movie_repo.get_all()
    
    def get_movie_by_id(self, movie_id: int) -> Optional[MovieInDB]:
        return self.movie_repo.get_by_id(movie_id)
    
    def create_movie(self, movie: MovieCreate) -> MovieInDB:
        result = self.movie_repo.create(movie)
        self._rebuild_vectors()
        return result
    
    def update_movie(self, movie_id: int, movie: MovieUpdate) -> Optional[MovieInDB]:
        result = self.movie_repo.update(movie_id, movie)
        if result:
            self._rebuild_vectors()
        return result
    
    def delete_movie(self, movie_id: int) -> bool:
        result = self.movie_repo.delete(movie_id)
        if result:
            self._rebuild_vectors()
        return result
    
    def _rebuild_vectors(self):
        """Rebuild vector index after CRUD operations"""
        try:
            corpus = self.movie_repo.get_corpus()
            if not corpus:
                return
            
            # Generate TF-IDF
            vocab_all = create_vocab_all(corpus)
            tf_all = compute_tf_all(vocab_all)
            self._idf_dict = compute_idf_single(corpus)
            tfidf_all = compute_tfidf_all(tf_all, self._idf_dict)
            
            # Get unique words for vector dimension
            unique_words = set()
            for doc in corpus:
                vocab = create_vocab_single(doc)
                unique_words.update(vocab.keys())
            self._unique_words = sorted(list(unique_words))
            
            # Convert to fixed dimension vectors
            vectors = []
            for tfidf_dict in tfidf_all:
                vector = [0.0] * len(self._unique_words)
                word_to_index = {word: i for i, word in enumerate(self._unique_words)}
                for word, value in tfidf_dict.items():
                    if word in word_to_index:
                        vector[word_to_index[word]] = value
                vectors.append(vector)
            
            # Rebuild collection
            self.vector_repo.create_collection(len(self._unique_words))
            
            # Prepare metadata
            movies = self.movie_repo.get_all()
            metadata = [
                {"movie_id": movie.id, "text": f"{movie.title} | {movie.genres}"}
                for movie in movies
            ]
            
            self.vector_repo.insert_vectors(vectors, metadata)
            logger.info("Vectors rebuilt successfully")
        except Exception as e:
            logger.error(f"Failed to rebuild vectors: {e}")
            raise
    
    def initialize(self):
        """Initialize the service with data"""
        try:
            corpus = self.movie_repo.get_corpus()
            if corpus:
                self._rebuild_vectors()
            logger.info("Movie service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize movie service: {e}")
            raise