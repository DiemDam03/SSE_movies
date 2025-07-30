import os
import psycopg2
from psycopg2.extras import RealDictCursor
from pymilvus import connections
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseConnections:
    _postgres_conn: Optional[psycopg2.extensions.connection] = None
    _milvus_connected: bool = False
    
    @classmethod
    def get_postgres_connection(cls):
        """Get PostgreSQL connection with connection pooling"""
        try:
            return psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", 5432),
                dbname=os.getenv("POSTGRES_DB", "movies"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "password")
            )
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    @classmethod
    def connect_to_milvus(cls):
        """Connect to Milvus"""
        if not cls._milvus_connected:
            try:
                connections.connect(
                    alias="default",
                    host=os.getenv("MILVUS_HOST", "localhost"),
                    port=os.getenv("MILVUS_PORT", "19530")
                )
                cls._milvus_connected = True
                logger.info("Connected to Milvus")
            except Exception as e:
                logger.error(f"Failed to connect to Milvus: {e}")
                raise
        return connections