from pymilvus import DataType, Collection, connections, CollectionSchema, FieldSchema, utility
import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import tfidf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MILVUS_DATABASE_PATH = os.path.join(BASE_DIR, "milvus.db")
CSV_PATH = os.path.join(BASE_DIR, "movies.csv")

COLLECTION_NAME = "movie_search"

def connect_to_milvus():
    return connections.connect(
        alias = "default",
        host = "localhost",
        port = "19530"
    ) 

def connect_to_postgres():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "movies"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "password")
    )

def create_postgres_table():
    conn = connect_to_postgres()
    cursor = conn.cursor()    

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            movieId INTEGER PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            genres VARCHAR(200)
        )
    ''')

    conn.commit()
    cursor.close()
    conn.close()

def create_milvus_collection():
    connect_to_milvus()
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1000)
        ]
    schema = CollectionSchema(fields, "Movies collection")

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    
    collection = Collection(COLLECTION_NAME, schema)

    # index_params = {
    #     "metric_type": "COSINE",
    #     "index_type": "IVF_FLAT",
    #     "params": {"nlist": 128}
    #     }
    
    # collection.create_index("vector", index_params)

    return collection

def convert_csv_to_postgres():
    df = pd.read_csv(CSV_PATH)
    conn = connect_to_postgres()
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

#not finish
def movie_dataset_processing_from_postgres():
    conn = connect_to_postgres()
    cursor = conn.cursor()
    # df = 
    pass

def generate_tfidf():
    dataset = movie_dataset_processing_from_postgres()
    vocab_all = tfidf.create_vocab_all(dataset)
    tf_all = tfidf.compute_tf_all(vocab_all)
    idf_dict = tfidf.compute_idf_single(dataset)
    tfidf_all = tfidf.compute_tfidf_all(tf_all, idf_dict)
    
    return dataset, tfidf_all, idf_dict

def converting_tfidf_to_fixed_vector():

    pass

def storing_vector_to_milvus():
    connect_to_milvus()

    pass    

def storing_metadata_to_postgres():
    pass

def loading_metadata_from_postgres():
    pass

def get_all_movies():
    conn = connect_to_postgres()
    cursor = conn.cursor()
    cursor.execute("SELECT movieId, title, genres FROM movies")
    movies = [{"id": row[0], "title": row[1], "genres": row[2]} for row in cursor.fetchall()]

    cursor.close()
    conn.close()
    return movies

def get_movie_by_id(movie_id: int):
    conn = connect_to_postgres()
    cursor = conn.cursor()

    cursor.execute("SELECT movieId, title, genres FROM movies WHERE movieId = ?", (movie_id,))
    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if row:
        return {"id": row[0], "title": row[1], "genres": row[2]}
    return None

def insert_movie(movie: dict):
    conn = connect_to_postgres()
    cursor = conn.cursor()

    cursor.execute("INSERT INTO movies (movieId, title, genres) VALUES (?, ?, ?)", 
              (movie["id"], movie["title"], movie["genres"]))
              
    conn.commit()
    cursor.close()
    conn.close()

def update_movie(movie_id: int, movie: dict):
    conn = connect_to_postgres()
    cursor = conn.cursor()

    cursor.execute("UPDATE movies SET title = ?, genres = ? WHERE movieId = ?", 
              (movie["title"], movie["genres"], movie_id))
    
    conn.commit()
    cursor.close()
    conn.close()

def delete_movie(movie_id: int):
    conn = connect_to_postgres()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM movies WHERE movieId = ?", (movie_id,))
    
    conn.commit()
    cursor.close()
    conn.close()


