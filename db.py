import os
import pandas as pd
import tfidf
import sqlite3
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DATASET_PATH = os.path.join(BASE_DIR, "movies.csv")
SQL_DB_PATH = os.path.join(BASE_DIR, "movies.sqlite")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db.sqlite")


import psycopg2
from psycopg2.extras import RealDictCursor
import os

def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "movies"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "yourpassword")
    )

import pandas as pd
import psycopg2

def migrate_sqlite_to_postgres():
    # đọc từ SQLite
    sqlite_df = pd.read_sql("SELECT * FROM movies", sqlite3.connect("movies.sqlite"))
    
    # ghi vào PostgreSQL
    conn = get_connection()
    cursor = conn.cursor()
    for _, row in sqlite_df.iterrows():
        cursor.execute("INSERT INTO movies (movieId, title, genres) VALUES (%s, %s, %s)",
                       (row["movieId"], row["title"], row["genres"]))
    conn.commit()
    conn.close()

def convert_csv_to_sqlite(csv_path, db_path):
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    df.to_sql("movies", conn, if_exists="replace", index=False)
    conn.close()

def movie_dataset_processing_from_sqlite():
    conn = sqlite3.connect(SQL_DB_PATH)
    df = pd.read_sql_query("SELECT title, genres FROM movies", conn)    
    df['text'] = df['title'] + ' | ' + df['genres']
    conn.close()
    return df['text'].tolist()

def generate_vectors():
    dataset = movie_dataset_processing_from_sqlite()
    vocab_all = tfidf.create_vocab_all(dataset)
    tf_all = tfidf.compute_tf_all(vocab_all)
    idf_dict = tfidf.compute_idf_single(dataset)
    tfidf_all = tfidf.compute_tfidf_all(tf_all, idf_dict)
    return dataset, tfidf_all

def storing_vectors(dataset: list[str], vectorized_dataset: list[dict]):
    conn = sqlite3.connect(VECTOR_DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            vector TEXT
        )
    ''')
    c.execute('DELETE FROM vectors')  

    for text, vector in zip(dataset, vectorized_dataset):
        vector_json = json.dumps(vector)
        c.execute('INSERT INTO vectors (text, vector) VALUES (?, ?)', (text, vector_json))

    conn.commit()
    conn.close()

def loading_vectors() -> list[dict]:
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError("Database not found. Please initialize it first.")

    conn = sqlite3.connect(VECTOR_DB_PATH)
    c = conn.cursor()
    c.execute('SELECT text, vector FROM vectors')
    rows = c.fetchall()
    conn.close()

    vector_store = []
    for text, vector_json in rows:
        vector_store.append({
            "text": text,
            "vector": json.loads(vector_json)
        })
    return vector_store

def initialize_database():
    if not os.path.exists(VECTOR_DB_PATH):
        print("Creating vector database...")
        dataset, vectorized_dataset = generate_vectors()
        storing_vectors(dataset, vectorized_dataset)
        print("Vector database created successfully!")
    else:
        print("Vector database already exists.")


def get_all_movies():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT movieId, title, genres FROM movies")
    movies = [{"id": row[0], "title": row[1], "genres": row[2]} for row in c.fetchall()]
    conn.close()
    return movies

def get_movie_by_id(movie_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT movieId, title, genres FROM movies WHERE movieId = ?", (movie_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return {"id": row[0], "title": row[1], "genres": row[2]}
    return None

def insert_movie(movie: dict):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO movies (movieId, title, genres) VALUES (?, ?, ?)", 
              (movie["id"], movie["title"], movie["genres"]))
    conn.commit()
    conn.close()

def update_movie(movie_id: int, movie: dict):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE movies SET title = ?, genres = ? WHERE movieId = ?", 
              (movie["title"], movie["genres"], movie_id))
    conn.commit()
    conn.close()

def delete_movie(movie_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM movies WHERE movieId = ?", (movie_id,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    convert_csv_to_sqlite(CSV_DATASET_PATH, SQL_DB_PATH)
    initialize_database()


