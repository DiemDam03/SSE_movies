import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pymilvus import DataType, Collection, connections, CollectionSchema, FieldSchema, utility
from repositories.interfaces.vector_repo import VectorREPO
from repositories.concrete.postgres_repo import PostgresREPO
from models.movie_model import Movie
from models.search_result_model import SearchResult
import core.tfidf as tfidf
from core.utilities import VectorHandler

COLLECTION_NAME = "movie_collection"

class MilvusREPO(VectorREPO):
    def __init__(self) -> None:
        self.collection_name = COLLECTION_NAME
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = int(os.getenv("MILVUS_PORT", "19530"))
        self.vec_handler = VectorHandler()
        self.postgres_repo = PostgresREPO()
        self.idf_dict = None
        self.unique_words = None

    def connect_to_milvus(self) -> None:
        try:
            return connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
        except Exception as e:
            print(f"Failed to connect to Milvus at {self.host}:{self.port}. Error: {e}")
            raise

    def create_schema(self, vector_dim: int) -> None: # return ko phải none, return schema, type của schema là gì
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="movieId", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim) # có thể cộng một số buffer(?) vào vec dim để chừa chỗ trống 
                                                                                    # cho add phim mới vào?
        ]
        schema = CollectionSchema(fields, "Movies collection")
        return schema

    def create_collection(self, vector_dim: int) -> None:
        self.connect_to_milvus()

        schema = self.create_schema(vector_dim)

        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        collection = Collection(self.collection_name, schema)
        return collection

    def store_vectors_to_milvus(self, vectors: list[list[float]], movie_data: list[dict]) -> None:        
        self.connect_to_milvus()
        collection = Collection(self.collection_name)  
        
        ids = list(range(len(vectors)))
        movie_ids = [data['id'] for data in movie_data]
        texts = [f"{data['title']} | {data['genres'] or ''}" for data in movie_data]
        
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

    def load_vectors_from_milvus(self) -> list[Movie]:
        self.connect_to_milvus()
        collection = Collection(self.collection_name)
        collection.load()
        
        all_results = []
        batch_size = 50
        total = collection.num_entities

        for offset in range(0, total, batch_size):
            expr = f"id >= {offset} and id < {offset + batch_size}"
            results = collection.query(expr=expr, output_fields=["id", "movieId", "text", "vector"])
            all_results.extend(results)

        return all_results
    
    def get_next_available_id(self) -> int:
        self.connect_to_milvus()
        collection = Collection(self.collection_name)
        collection.load()
        if collection.num_entities == 0:
            return 0
        results = collection.query(expr="id >= 0", output_fields=["id"])
        if not results:
            return 0
        max_id = max(result["id"] for result in results)
        return max_id + 1
    
    def vectorize_movie_text(self, movie_text: str, corpus: list[str]) -> list[float]:
        if self.unique_words is None or self.idf_dict is None:
            self.refresh_collection_state()

        movie_vocab = tfidf.create_vocab_single(movie_text)
        movie_tf = tfidf.compute_tf_single(movie_vocab)
        movie_tfidf = tfidf.compute_tfidf_single(movie_tf, self.idf_dict)
        movie_vector = self.vec_handler.converting_tfidf_to_fixed_dim_vector(movie_tfidf, self.unique_words)    

        return movie_vector
    
    def add_movie(self, movie_data: dict) -> None:
        self.connect_to_milvus()
        collection = Collection(self.collection_name)
        collection.load() # ko liên quan, ko gây lỗi

        movie_text = f"{movie_data['title']} | {movie_data.get('genres', '') or ''}" # ko liên quan, ko gây lỗi
        movie_vector = self.vectorize_movie_text() # khả năng gây lỗi 
        # cần update collection
        milvus_id = self.get_next_available_id() # ko liên quan, ko gây lỗi

        entities = [
            [milvus_id],  
            [movie_data['id']],  
            [movie_text],  
            [movie_vector]  
        ] # ko liên quan, ko gây lỗi

        collection.insert(entities) 
        collection.flush() 

    def update_movie(self, movie_id: int, movie_data: dict, corpus: list[str]) -> None:
        self.connect_to_milvus()
        collection = Collection(self.collection_name)
        collection.load()

        # self.delete_movie(movie_id)
        # self.add_movie(movie_data, corpus)
        # or? what the different?
        collection.upsert(movie_data)
        collection.flush()

    def delete_movie(self, movie_id: int) -> None:
        self.connect_to_milvus()
        collection = Collection(self.collection_name)
        collection.load()

        expr = f"movieId == {movie_id}"
        collection.delete(expr)
        collection.flush()

    def search_top_k_movie(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        self.connect_to_milvus()
        collection = Collection(self.collection_name)
        collection.load()

        search_results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["id", "movieId", "text"]
        )

        top_hits = search_results[0]
        final_results = [
            SearchResult(index=hit.entity.get("movieId"), score=hit.distance, text=hit.entity.get("text"))
            for hit in top_hits
        ]
        return final_results
    
    def refresh_collection_state(self) -> None:
        new_corpus = self.postgres_repo.get_corpus()
        _, self.idf_dict = self.vec_handler.generate_tfidf(new_corpus)
        self.unique_words = self.vec_handler.get_unique_words(new_corpus)
