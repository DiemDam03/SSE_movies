import os
import pandas as pd
from typing import Optional
from repositories.movie_repository import MovieRepositoryInterface
from repositories.vector_repository import VectorRepositoryInterface
from models.movie import MovieCreate
import logging
from core.tfidf import *


logger = logging.getLogger(__name__)

class InitializationService:
    def __init__(self, movie_repo: MovieRepositoryInterface, vector_repo: VectorRepositoryInterface):
        self.movie_repo = movie_repo
        self.vector_repo = vector_repo
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.csv_path = os.path.join(self.base_dir, "movies.csv")
    
    def initialize_database(self):
        """Initialize the entire database system"""
        try:
            logger.info("Starting database initialization...")
            
            # Check if movies already exist
            existing_movies = self.movie_repo.get_all()
            
            if not existing_movies:
                logger.info("No movies found, loading from CSV...")
                self.load_movies_from_csv()
            else:
                logger.info(f"Found {len(existing_movies)} existing movies")
            
            # Initialize vectors
            self._initialize_vectors()
            
            logger.info("Database initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def load_movies_from_csv(self):
        """Load movies from CSV file into PostgreSQL"""
        if not os.path.exists(self.csv_path):
            logger.warning(f"CSV file not found: {self.csv_path}")
            logger.info("Creating sample data instead...")
            self._create_sample_data()
            return
        
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loading {len(df)} movies from CSV...")
            
            for _, row in df.iterrows():
                movie = MovieCreate(
                    id=int(row['movieId']),
                    title=str(row['title']),
                    genres=str(row.get('genres', ''))
                )
                try:
                    self.movie_repo.create(movie)
                except Exception as e:
                    logger.warning(f"Failed to insert movie {movie.id}: {e}")
            
            logger.info("CSV data loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def _create_sample_data(self):
        """Create sample movies if CSV is not available"""
        sample_movies = [
            MovieCreate(id=1, title="Toy Story", genres="Adventure|Animation|Children|Comedy|Fantasy"),
            MovieCreate(id=2, title="Jumanji", genres="Adventure|Children|Fantasy"),
            MovieCreate(id=3, title="Grumpier Old Men", genres="Comedy|Romance"),
            MovieCreate(id=4, title="Waiting to Exhale", genres="Comedy|Drama|Romance"),
            MovieCreate(id=5, title="Father of the Bride Part II", genres="Comedy"),
            MovieCreate(id=6, title="Heat", genres="Action|Crime|Thriller"),
            MovieCreate(id=7, title="Sabrina", genres="Comedy|Romance"),
            MovieCreate(id=8, title="Tom and Huck", genres="Adventure|Children"),
            MovieCreate(id=9, title="Sudden Death", genres="Action"),
            MovieCreate(id=10, title="GoldenEye", genres="Action|Adventure|Thriller")
        ]
        
        logger.info("Creating sample movie data...")
        for movie in sample_movies:
            try:
                self.movie_repo.create(movie)
            except Exception as e:
                logger.warning(f"Failed to create sample movie {movie.id}: {e}")
        
        logger.info("Sample data created successfully")
    
    def _initialize_vectors(self):
        """Initialize vector database"""        
        try:
            corpus = self.movie_repo.get_corpus()
            if not corpus:
                logger.warning("No corpus available for vector initialization")
                return
            
            logger.info("Initializing vectors...")
            
            # Generate TF-IDF
            vocab_all = create_vocab_all(corpus)
            tf_all = compute_tf_all(vocab_all)
            idf_dict = compute_idf_single(corpus)
            tfidf_all = compute_tfidf_all(tf_all, idf_dict)
            
            # Get unique words for vector dimension
            unique_words = set()
            for doc in corpus:
                vocab = create_vocab_single(doc)
                unique_words.update(vocab.keys())
            unique_words = sorted(list(unique_words))
            
            # Convert to fixed dimension vectors
            vectors = []
            for tfidf_dict in tfidf_all:
                vector = [0.0] * len(unique_words)
                word_to_index = {word: i for i, word in enumerate(unique_words)}
                for word, value in tfidf_dict.items():
                    if word in word_to_index:
                        vector[word_to_index[word]] = value
                vectors.append(vector)
            
            # Create collection and insert vectors
            self.vector_repo.create_collection(len(unique_words))
            
            # Prepare metadata
            movies = self.movie_repo.get_all()
            metadata = [
                {"movie_id": movie.id, "text": f"{movie.title} | {movie.genres}"}
                for movie in movies
            ]
            
            self.vector_repo.insert_vectors(vectors, metadata)
            logger.info(f"Initialized {len(vectors)} vectors with dimension {len(unique_words)}")
            
        except Exception as e:
            logger.error(f"Vector initialization failed: {e}")
            raise
    
    def reset_database(self):
        """Reset the entire database - USE WITH CAUTION"""
        logger.warning("Resetting database - all data will be lost!")
        
        try:
            # Delete all movies (this should cascade)
            movies = self.movie_repo.get_all()
            for movie in movies:
                self.movie_repo.delete(movie.id)
            
            # Delete vector collection
            self.vector_repo.delete_collection()
            
            logger.info("Database reset completed")
            
        except Exception as e:
            logger.error(f"Database reset failed: {e}")
            raise