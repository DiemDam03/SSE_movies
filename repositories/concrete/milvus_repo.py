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
        self.idf_dict = {}
        self.unique_words = []

    def connect_to_milvus(self) -> None:
        try:
            return connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )
        except Exception as e:
            print(f"Failed to connect to Milvus at {self.host}:{self.port}. Error: {e}")
            raise

    def create_collection(self, vector_dim: int) -> Collection:
        self.connect_to_milvus()

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="movieId", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)

        ]
        schema = CollectionSchema(fields, "Movies collection")

        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        collection = Collection(self.collection_name, schema)
        return collection 
        
    def check_if_need_renew_colleciton(self, movie_text: str, current_unique_words: list[str]) -> bool:
        if not current_unique_words:
            self.refresh_collection_state()
            
        current_vocab = set(current_unique_words)

        that_movie_vocab = set(tfidf.create_vocab_single(movie_text).keys())
        
        return not that_movie_vocab.issubset(current_vocab)

    def rebuild_collection(self) -> None:
        movie_data = self.postgres_repo.get_all_movies()
        if not movie_data:
            return {"message": "Emty database"}

        new_corpus = [f"{row['title']} | {row['genres'] or ''}" for row in movie_data]

        new_corpus_unique_words = self.vec_handler.get_unique_words(new_corpus)

        if not self.unique_words: 
            self.unique_words = new_corpus_unique_words
        else:
            self.unique_words = sorted(list(set(self.unique_words) | set(new_corpus_unique_words))) 
        tfidf_all, self.idf_dict = self.vec_handler.generate_tfidf(new_corpus)

        modified_vectors =[
            self.vec_handler.converting_tfidf_to_fixed_dim_vector(tfidf_dict, self.unique_words)
            for tfidf_dict in tfidf_all
        ]                                  

        self.create_collection(len(self.unique_words))

        self.store_vectors_to_milvus(modified_vectors, movie_data)

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

    def refresh_collection_state(self) -> None:

        movie_data = self.postgres_repo.get_all_movies()
        new_corpus = [f"{row['title']} | {row['genres'] or ''}" for row in movie_data]
        new_corpus_unique_words = self.vec_handler.get_unique_words(new_corpus)
        if not self.unique_words: 
            self.unique_words = new_corpus_unique_words
        else:
            self.unique_words = sorted(list(set(self.unique_words) | set(new_corpus_unique_words)))
        _, self.idf_dict = self.vec_handler.generate_tfidf(new_corpus)

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
    
    def vectorize_movie_text(self, movie_text: str) -> list[float]:
        if self.unique_words is None or self.idf_dict is None:
            self.refresh_collection_state() 

        movie_vocab = tfidf.create_vocab_single(movie_text)
        movie_tf = tfidf.compute_tf_single(movie_vocab)
        movie_tfidf = tfidf.compute_tfidf_single(movie_tf, self.idf_dict)
        movie_vector = self.vec_handler.converting_tfidf_to_fixed_dim_vector(movie_tfidf, self.unique_words)    

        return movie_vector
    
    def add_movie(self, movie_data: dict) -> None:
        if self.unique_words is None or self.idf_dict is None:
            self.refresh_collection_state()

        self.connect_to_milvus()
        collection = Collection(self.collection_name)
        collection.load() 

        movie_text = f"{movie_data['title']} | {movie_data.get('genres', '') or ''}" 

        
        movie_vector = self.vectorize_movie_text(movie_text) 
        milvus_id = self.get_next_available_id() 
        if self.check_if_need_renew_colleciton(movie_text, self.unique_words):
                    self.rebuild_collection()

        entities = [
            [milvus_id],  
            [movie_data['id']],  
            [movie_text],  
            [movie_vector]  
        ]   

        collection.insert(entities) 
        collection.flush() 

    def update_movie(self, movie_id: int , movie_data: dict) -> None:
        if self.unique_words is None or self.idf_dict is None:
            self.refresh_collection_state()
        
        self.connect_to_milvus()
        collection = Collection(self.collection_name)
        collection.load()

        movie_text = f"{movie_data['title']} | {movie_data.get('genres', '') or ''}" 

        expr = f"movieId == {movie_id}"
        existing_records = collection.query(expr=expr, output_fields=["id", "movieId"])
    
        if not existing_records:
            self.add_movie(movie_data)  
            return
        
        milvus_id = existing_records[0]["id"]

        movie_vector = self.vectorize_movie_text(movie_text) 

        if self.check_if_need_renew_colleciton(movie_text, self.unique_words):
            self.rebuild_collection()

        entities = [
            [milvus_id],              
            [movie_data['id']],      
            [movie_text],
            [movie_vector]
        ]

        collection.upsert(entities) 
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
