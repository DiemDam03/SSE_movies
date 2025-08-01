import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from repositories.interfaces.movie_repo import MovieREPO
from models.movie_model import Movie
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd

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

    def transfer_data_from_csv_to_postgres(self, csv_path) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
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

    def loading_data_from_postgres(self) -> list[str]:
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

    def update_postgres(): #cần để update unique words? idf_dict? wait unique word với idf dict là của search, 
                            #nó có là của milvus, lưu vào milvus?
        pass

    def get_all_movies(self) -> list[Movie]:
        conn = self.connect_to_postgres()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT movieId, title, genres FROM movies ORDER BY movieId")
        movies = [{"id": row['movieid'], "title": row['title'], "genres": row['genres']} 
                 for row in cursor.fetchall()]

        cursor.close()
        conn.close()
        return movies
    
    def get_movie_by_id(self, movie_id: int) -> Movie:
        conn = self.connect_to_postgres()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT movieId, title, genres FROM movies WHERE movieId = %s", (movie_id,))
        row = cursor.fetchone()

        cursor.close()
        conn.close()

        if row:
            return {"id": row['movieid'], "title": row['title'], "genres": row['genres']}
        return None
    
    def add_movie(self, movie: dict) -> None:
        conn = self.connect_to_postgres()
        cursor = conn.cursor()

        cursor.execute("INSERT INTO movies (movieId, title, genres) VALUES (%s, %s, %s)", 
                      (movie["id"], movie["title"], movie["genres"]))
                      
        conn.commit()
        cursor.close()
        conn.close()

    def update_movie(self, movie_id: int, movie: dict) -> None:
        conn = self.connect_to_postgres()
        cursor = conn.cursor()

        cursor.execute("UPDATE movies SET title = %s, genres = %s WHERE movieId = %s", 
                      (movie["title"], movie["genres"], movie_id))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def delete_movie(self, movie_id: int) -> None:
        conn = self.connect_to_postgres()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM movies WHERE movieId = %s", (movie_id,))
        
        conn.commit()
        cursor.close()
        conn.close()


