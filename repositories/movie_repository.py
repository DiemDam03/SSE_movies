from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from models.movie import Movie, MovieCreate, MovieUpdate, MovieInDB
from database.connections import DatabaseConnections
from psycopg2.extras import RealDictCursor
import logging

logger = logging.getLogger(__name__)

class MovieRepositoryInterface(ABC):
    @abstractmethod
    def get_all(self) -> List[MovieInDB]:
        pass
    
    @abstractmethod
    def get_by_id(self, movie_id: int) -> Optional[MovieInDB]:
        pass
    
    @abstractmethod
    def create(self, movie: MovieCreate) -> MovieInDB:
        pass
    
    @abstractmethod
    def update(self, movie_id: int, movie: MovieUpdate) -> Optional[MovieInDB]:
        pass
    
    @abstractmethod
    def delete(self, movie_id: int) -> bool:
        pass
    
    @abstractmethod
    def get_corpus(self) -> List[str]:
        pass

class PostgresMovieRepository(MovieRepositoryInterface):
    def __init__(self):
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Create table if it doesn't exist"""
        try:
            conn = DatabaseConnections.get_postgres_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS movies (
                    movieId INTEGER PRIMARY KEY,
                    title VARCHAR(500) NOT NULL,
                    genres VARCHAR(200),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            ''')
            
            cursor.execute('''
                DROP TRIGGER IF EXISTS update_movies_updated_at ON movies;
                CREATE TRIGGER update_movies_updated_at
                    BEFORE UPDATE ON movies
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            ''')

            conn.commit()
            cursor.close()
            conn.close()
            logger.info("PostgreSQL table initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL table: {e}")
            raise
    
    def get_all(self) -> List[MovieInDB]:
        try:
            conn = DatabaseConnections.get_postgres_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM movies ORDER BY movieId")
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [MovieInDB(**dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get all movies: {e}")
            raise
    
    def get_by_id(self, movie_id: int) -> Optional[MovieInDB]:
        try:
            conn = DatabaseConnections.get_postgres_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM movies WHERE movieId = %s", (movie_id,))
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return MovieInDB(**dict(row)) if row else None
        except Exception as e:
            logger.error(f"Failed to get movie {movie_id}: {e}")
            raise
    
    def create(self, movie: MovieCreate) -> MovieInDB:
        try:
            conn = DatabaseConnections.get_postgres_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(
                "INSERT INTO movies (movieId, title, genres) VALUES (%s, %s, %s) RETURNING *",
                (movie.id, movie.title, movie.genres)
            )
            row = cursor.fetchone()
            conn.commit()
            cursor.close()
            conn.close()
            
            return MovieInDB(**dict(row))
        except Exception as e:
            logger.error(f"Failed to create movie: {e}")
            raise
    
    def update(self, movie_id: int, movie: MovieUpdate) -> Optional[MovieInDB]:
        try:
            conn = DatabaseConnections.get_postgres_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(
                "UPDATE movies SET title = %s, genres = %s WHERE movieId = %s RETURNING *",
                (movie.title, movie.genres, movie_id)
            )
            row = cursor.fetchone()
            conn.commit()
            cursor.close()
            conn.close()
            
            return MovieInDB(**dict(row)) if row else None
        except Exception as e:
            logger.error(f"Failed to update movie {movie_id}: {e}")
            raise
    
    def delete(self, movie_id: int) -> bool:
        try:
            conn = DatabaseConnections.get_postgres_connection()
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM movies WHERE movieId = %s", (movie_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            cursor.close()
            conn.close()
            
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete movie {movie_id}: {e}")
            raise
    
    def get_corpus(self) -> List[str]:
        try:
            conn = DatabaseConnections.get_postgres_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT movieId, title, genres FROM movies ORDER BY movieId")
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [f"{row['title']} | {row['genres'] or ''}" for row in rows]
        except Exception as e:
            logger.error(f"Failed to get corpus: {e}")
            raise
