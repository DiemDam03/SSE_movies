import logging
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from repositories.movie_repository import PostgresMovieRepository
from repositories.vector_repository import MilvusVectorRepository
from services.initialization_service import InitializationService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main initialization function"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting database initialization...")
        
        # Initialize repositories
        movie_repo = PostgresMovieRepository()
        vector_repo = MilvusVectorRepository()
        
        # Initialize the service
        init_service = InitializationService(movie_repo, vector_repo)
        
        # Run initialization
        init_service.initialize_database()
        
        logger.info("Database initialization completed successfully!")
        logger.info("You can now start the API servers:")
        logger.info("  - CRUD API: python app_crud.py")
        logger.info("  - Search API: python app_search.py")
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)

def reset_database():
    """Reset database function"""
    logger = logging.getLogger(__name__)
    
    response = input("Are you sure you want to reset the database? This will delete all data! (yes/no): ")
    if response.lower() != 'yes':
        logger.info("Database reset cancelled")
        return
    
    try:
        logger.info("Resetting database...")
        
        movie_repo = PostgresMovieRepository()
        vector_repo = MilvusVectorRepository()
        init_service = InitializationService(movie_repo, vector_repo)
        
        init_service.reset_database()
        logger.info("Database reset completed!")
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database initialization script")
    parser.add_argument("--reset", action="store_true", help="Reset the database (deletes all data)")
    
    args = parser.parse_args()
    
    if args.reset:
        reset_database()
    else:
        main()