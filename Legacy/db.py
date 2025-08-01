from pymilvus import DataType, Collection, connections, CollectionSchema, FieldSchema, utility
import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import dependencies.tfidf as tfidf
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "movies.csv")

COLLECTION_NAME = "movie_collection"

class DatabaseManager:
    def __init__(self):
        self.idf_dict = None  
        self.unique_words = None  
        self._corpus_cache = None  

    @staticmethod
    def connect_to_milvus():
        return connections.connect(
            alias="default",
            host="localhost",
            port=19530
        ) 

    @staticmethod
    def connect_to_postgres():
        return psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", 5432),
            dbname=os.getenv("POSTGRES_DB", "movies"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "password")
        )

    def create_postgres_table(self):
        conn = self.connect_to_postgres()
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

    def create_milvus_collection(self):
        self.connect_to_milvus()
        
        if self.unique_words is None:
            corpus = self.movie_dataset_processing_from_postgres()
            self.unique_words = self.get_unique_words_for_vector_dim(corpus)
        
        dim = len(self.unique_words)
        
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

    def invalidate_cache(self):
        self.idf_dict = None
        self.unique_words = None
        self._corpus_cache = None

    def update_dict_after_crud(self):
        self.invalidate_cache()
        corpus = self.movie_dataset_processing_from_postgres()
        _, _, self.idf_dict = self.generate_tfidf()
        self.unique_words = self.get_unique_words_for_vector_dim(corpus)

    def sync_postgres_and_milvus(self):
        self.update_dict_after_crud() # sao lại update trước khi sync
        
        self.connect_to_milvus()
        if utility.has_collection(COLLECTION_NAME):
            collection = Collection(COLLECTION_NAME)
            collection.drop() # drop hoàn toàn ko add vào thêm? cost nhiều
        
        self.create_milvus_collection() 
        self.storing_vectors()

    def movie_dataset_processing_from_postgres(self):
        conn = self.connect_to_postgres()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT movieId, title, genres FROM movies ORDER BY movieId")
        rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        texts = []
        for row in rows:
            text = f"{row['title']} | {row['genres'] or ''}"
            texts.append(text)
        
        return texts

    def generate_tfidf(self):
        dataset = self.movie_dataset_processing_from_postgres()
        vocab_all = tfidf.create_vocab_all(dataset)
        tf_all = tfidf.compute_tf_all(vocab_all)
        idf_dict = tfidf.compute_idf_single(dataset)
        tfidf_all = tfidf.compute_tfidf_all(tf_all, idf_dict)
        return dataset, tfidf_all, idf_dict

    def get_unique_words_for_vector_dim(self, corpus: list[str])->list[str]:
        unique_words = set()
        for doc in corpus:
            vocab = tfidf.create_vocab_single(doc)
            unique_words.update(vocab.keys())  
        return sorted(list(unique_words))
    
    def converting_tfidf_to_fixed_dim_vector(self, tfidf_dict: dict, unique_words: list[str])->list[float]:
        vector = [0.0] * len(unique_words)
        word_to_index = {word: i for i, word in enumerate(unique_words)}
        for word, value in tfidf_dict.items():
            if word in word_to_index:
                vector[word_to_index[word]] = value
        return vector

    def generate_vectors(self, corpus: list[str]):
        _, tfidf_all, _ = self.generate_tfidf()
        unique_words = self.get_unique_words_for_vector_dim(corpus)
        modified_vectors = [
            self.converting_tfidf_to_fixed_dim_vector(tfidf_dict, unique_words)
            for tfidf_dict in tfidf_all
        ]
        return modified_vectors

    def storing_vectors(self):
        dataset = self.movie_dataset_processing_from_postgres()
        vectors = self.generate_vectors(dataset)
        
        conn = self.connect_to_postgres()
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

    def loading_vectors_from_milvus(self):
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

    def loading_metadata_from_postgres(self):
        conn = self.connect_to_postgres()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("SELECT movieId, title, genres FROM movies ORDER BY movieId")
        movies = {row['movieid']: {
            'title': row['title'],
            'genres': row['genres'] or '',
            'text': f"{row['title']} | {row['genres'] or ''}"
        } for row in cursor.fetchall()}
        
        cursor.close()
        conn.close()
        return movies

    def initialize_database(self):
        self.create_postgres_table()
        
        if not self.get_all_movies():
            self.convert_csv_to_postgres()
        
        self.create_milvus_collection()
        self.storing_vectors()

    def convert_csv_to_postgres(self):
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
            
        df = pd.read_csv(CSV_PATH)
        conn = self.connect_to_postgres()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM movies")

        for _, row in df.iterrows():
            cursor.execute(
                "INSERT INTO movies (movieId, title, genres) VALUES (%s, %s, %s)",
                (int(row['movieId']), row['title'], row.get('genres', ''))
            )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        self.invalidate_cache()

    def get_all_movies(self):
        conn = self.connect_to_postgres()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT movieId, title, genres FROM movies ORDER BY movieId")
        movies = [{"id": row['movieid'], "title": row['title'], "genres": row['genres']} 
                 for row in cursor.fetchall()]

        cursor.close()
        conn.close()
        return movies

    def get_movie_by_id(self, movie_id: int):
        conn = self.connect_to_postgres()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("SELECT movieId, title, genres FROM movies WHERE movieId = %s", (movie_id,))
        row = cursor.fetchone()

        cursor.close()
        conn.close()

        if row:
            return {"id": row['movieid'], "title": row['title'], "genres": row['genres']}
        return None

    def insert_movie(self, movie: dict):
        conn = self.connect_to_postgres()
        cursor = conn.cursor()

        cursor.execute("INSERT INTO movies (movieId, title, genres) VALUES (%s, %s, %s)", 
                      (movie["id"], movie["title"], movie["genres"]))
                      
        conn.commit()
        cursor.close()
        conn.close()
        
        self.sync_postgres_and_milvus()

    def update_movie(self, movie_id: int, movie: dict):
        conn = self.connect_to_postgres()
        cursor = conn.cursor()

        cursor.execute("UPDATE movies SET title = %s, genres = %s WHERE movieId = %s", 
                      (movie["title"], movie["genres"], movie_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        self.sync_postgres_and_milvus()

    def delete_movie(self, movie_id: int):
        conn = self.connect_to_postgres()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM movies WHERE movieId = %s", (movie_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        self.sync_postgres_and_milvus()

dbm = DatabaseManager()

if __name__ == "__main__":
    dbm.initialize_database()