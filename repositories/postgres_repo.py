
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from repositories.movie_repo import MovieREPO
from typing import List, Optional
from psycopg2.extras import RealDictCursor
import psycopg2
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "movies.csv")

class PostgresREPO(MovieREPO):
    def __init__(self) -> None:
        self.connection_params = {
            'host': os.getenv("POSTGRES_HOST", "localhost"),
            'port': os.getenv("POSTGRES_PORT", 5432),
            'dbname': os.getenv("POSTGRES_DB", "movies"),
            'user': os.getenv("POSTGRES_USER", "postgres"),
            'password': os.getenv("POSTGRES_PASSWORD", "password")
        }

    def connect_to_postgres(self) -> None:
        return psycopg2.connect(**self.connection_params)
    
    def create_table(self) -> None:
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

    def transfer_data_from_csv_to_postgres(self):
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
        
        self.idf_dict = None
        self.unique_words = None
        self._corpus_cache = None

    # def save_to_postgres():
    #     pass

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

    def update_postgres():
        pass