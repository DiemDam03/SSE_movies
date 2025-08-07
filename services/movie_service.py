import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.movie_model import Movie
from repositories.concrete.milvus_repo import MilvusREPO
from repositories.concrete.postgres_repo import PostgresREPO
from services.data_service import DataManager

class MovieService:
    def __init__(self, postgres_repo=None, milvus_repo=None, data_manager=None) -> None:
        self.postgres_repo = postgres_repo if postgres_repo else PostgresREPO()
        self.milvus_repo = milvus_repo if milvus_repo else MilvusREPO()
        self.data_manager = data_manager if data_manager else DataManager()

    def get_all_movies(self) -> list[Movie]:
        return self.postgres_repo.get_all_movies()

    def get_movie_by_id(self, movie_id: int) -> Movie | None:
        movie = self.postgres_repo.get_movie_by_id(movie_id)
        return movie

    def add_movie(self, movie: dict) -> dict:
        try:
            # First add to PostgreSQL
            self.postgres_repo.add_movie(movie)
            
            # Get updated corpus after adding the movie
            corpus = self.postgres_repo.get_corpus()
            
            # Refresh Milvus collection state with new corpus
            self.milvus_repo.refresh_collection_state(corpus)
            
            # Add movie to Milvus
            self.milvus_repo.add_movie(movie, corpus)
            
            return {"message": f"Movie {movie['id']} added successfully to both PostgreSQL and Milvus"}
        except Exception as e:
            # If there's an error, we should rollback the PostgreSQL transaction
            try:
                self.postgres_repo.delete_movie(movie['id'])
            except:
                pass
            raise Exception(f"Failed to add movie: {str(e)}")
        
    def update_movie(self, movie_id: int, movie: dict) -> dict:
        try:
            # Check if movie exists
            existing_movie = self.postgres_repo.get_movie_by_id(movie_id)
            if not existing_movie:
                raise Exception(f"Movie with ID {movie_id} not found")
            
            # Update in PostgreSQL
            self.postgres_repo.update_movie(movie_id, movie)
            
            # Get updated corpus after updating the movie
            corpus = self.postgres_repo.get_corpus()
            
            # Refresh Milvus collection state with new corpus
            self.milvus_repo.refresh_collection_state(corpus)
            
            # Create updated movie data for Milvus
            updated_movie_data = {
                'id': movie_id,
                'title': movie['title'],
                'genres': movie['genres']
            }
            
            # Update movie in Milvus
            self.milvus_repo.update_movie(movie_id, updated_movie_data, corpus)
            
            return {"message": f"Movie {movie_id} updated successfully in both PostgreSQL and Milvus"}
        except Exception as e:
            raise Exception(f"Failed to update movie: {str(e)}")

    def delete_movie(self, movie_id: int) -> dict:
        try:
            # Check if movie exists
            existing_movie = self.postgres_repo.get_movie_by_id(movie_id)
            if not existing_movie:
                raise Exception(f"Movie with ID {movie_id} not found")
            
            # Delete from Milvus first
            self.milvus_repo.delete_movie(movie_id)
            
            # Delete from PostgreSQL
            self.postgres_repo.delete_movie(movie_id)
            
            # Get updated corpus after deleting the movie
            corpus = self.postgres_repo.get_corpus()
            
            # Refresh Milvus collection state with new corpus
            if corpus:  # Only refresh if there are still movies left
                self.milvus_repo.refresh_collection_state(corpus)
            
            return {"message": f"Movie {movie_id} deleted successfully from both PostgreSQL and Milvus"}
        except Exception as e:
            raise Exception(f"Failed to delete movie: {str(e)}")