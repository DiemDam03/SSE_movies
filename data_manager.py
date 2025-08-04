import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from repositories.concrete.postgres_repo import PostgresREPO
from repositories.concrete.milvus_repo import MilvusREPO
from core.utilities import VectorHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "movies.csv")

class DataManager:
    def __init__(self) -> None:
        self.pg_repo = PostgresREPO()
        self.mv_repo = MilvusREPO()
        self.vec_handler = VectorHandler()
    
    def init_data(self) -> None:
        self.pg_repo.create_table()

        self.pg_repo.transfer_data_from_csv_to_postgres(CSV_PATH)

        self.sync_postgres_milvus()
       
    def sync_postgres_milvus(self) -> None:
        corpus = self.pg_repo.get_corpus()
        vectors, unique_words = self.vec_handler.generate_vectors(corpus)

        self.mv_repo.create_collection(len(unique_words))


        movie_data = self.pg_repo.get_all_movies()
        self.mv_repo.store_vectors_to_milvus(vectors, movie_data)

if __name__ == "__main__":
    dm = DataManager()
    dm.init_data()