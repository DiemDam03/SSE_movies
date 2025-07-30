from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pymilvus import DataType, Collection, CollectionSchema, FieldSchema, utility
from database.connections import DatabaseConnections
import logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "movie_collection"

class VectorRepositoryInterface(ABC):
    @abstractmethod
    def create_collection(self, dimension: int) -> None:
        pass
    
    @abstractmethod
    def insert_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        pass
    
    @abstractmethod
    def search_vectors(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_all_vectors(self) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def delete_collection(self) -> None:
        pass

class MilvusVectorRepository(VectorRepositoryInterface):
    def __init__(self):
        DatabaseConnections.connect_to_milvus()
    
    def create_collection(self, dimension: int) -> None:
        try:
            if utility.has_collection(COLLECTION_NAME):
                utility.drop_collection(COLLECTION_NAME)
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="movieId", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            schema = CollectionSchema(fields, "Movies collection for TF-IDF search")
            Collection(COLLECTION_NAME, schema)
            logger.info(f"Created Milvus collection with dimension {dimension}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def insert_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        try:
            collection = Collection(COLLECTION_NAME)
            
            ids = list(range(len(vectors)))
            movie_ids = [item['movie_id'] for item in metadata]
            texts = [item['text'] for item in metadata]
            
            entities = [ids, movie_ids, texts, vectors]
            collection.insert(entities)
            collection.flush()
            logger.info(f"Inserted {len(vectors)} vectors")
        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            raise
    
    def search_vectors(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        try:
            collection = Collection(COLLECTION_NAME)
            collection.load()
            
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = collection.search(
                [query_vector], 
                "vector", 
                search_params, 
                limit=top_k,
                output_fields=["movieId", "text"]
            )
            
            search_results = []
            for hit in results[0]:
                search_results.append({
                    "movie_id": hit.entity.get("movieId"),
                    "text": hit.entity.get("text"),
                    "score": hit.score,
                    "id": hit.id
                })
            
            return search_results
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise
    
    def get_all_vectors(self) -> List[Dict[str, Any]]:
        try:
            collection = Collection(COLLECTION_NAME)
            collection.load()
            
            results = collection.query(
                expr="id >= 0",
                output_fields=["id", "movieId", "text", "vector"]
            )
            return results
        except Exception as e:
            logger.error(f"Failed to get all vectors: {e}")
            raise
    
    def delete_collection(self) -> None:
        try:
            if utility.has_collection(COLLECTION_NAME):
                utility.drop_collection(COLLECTION_NAME)
                logger.info("Deleted Milvus collection")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise