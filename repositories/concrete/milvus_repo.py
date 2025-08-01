import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pymilvus import DataType, Collection, connections, CollectionSchema, FieldSchema, utility
from repositories.interfaces.vector_repo import VectorREPO
from repositories.concrete.postgres_repo import PostgresREPO
from dependencies.utilities import utilities
from typing import List, Optional
from psycopg2.extras import RealDictCursor


COLLECTION_NAME = "movie_collection"

uti = utilities()
pg = PostgresREPO()

class MilvusREPO(VectorREPO):
    def __init__(self):
        self.collection_name = COLLECTION_NAME
        self.unique_words = None

    def connect_to_milvus():
        return connections.connect(
            alias="default",
            host="localhost",
            port=19530
        ) 

    def create_collection(self):
        self.connect_to_milvus()

        if self.unique_words is None: # cần xem lại cái unique word là của ai
            corpus = uti.movie_dataset_processing_from_postgres()
            unique_words = uti.get_unique_words_for_vector_dim(corpus)

        dim = len(unique_words)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="movieId", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields, "Movies collection for TF-IDF search")

        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)

        collection = Collection(COLLECTION_NAME, schema)
        return collection

    def store_vectors_to_milvus(self):
        dataset = utilities.movie_dataset_processing_from_postgres()
        vectors = utilities.generate_vectors(dataset)
        
        conn = pg.connect_to_postgres()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT movieId, title, genres FROM movies ORDER BY movieId")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        self.connect_to_milvus()
        collection = Collection(COLLECTION_NAME)
        
        ids = list(range(len(vectors)))
        movie_ids = [row['movieid'] for row in rows]
        texts = [f"{row['title']} | {row['genres'] or ''}" for row in rows]
        
        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            batch_entities = [
                ids[i:i+batch_size],
                movie_ids[i:i+batch_size],
                texts[i:i+batch_size],
                vectors[i:i+batch_size],
            ]
            collection.insert(batch_entities)

        collection.flush()

        collection.create_index(
            field_name="vector",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        )

    def load_vectors_from_milvus(self):
        self.connect_to_milvus()
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        all_results = []
        batch_size = 50
        total = collection.num_entities
        for offset in range(0, total, batch_size):
            expr = f"id >= {offset} and id < {offset + batch_size}"
            results = collection.query(expr=expr, output_fields=["id", "movieId", "text", "vector"])
            all_results.extend(results)
        return all_results
