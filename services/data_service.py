import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from repositories.concrete.postgres_repo import PostgresREPO
from repositories.concrete.milvus_repo import MilvusREPO
from core.utilities import VectorHandler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "movies.csv")

class DataManager:
    def __init__(self, postgres_repo=None, milvus_repo=None, vec_handler=None) -> None:
        self.postgres_repo = postgres_repo if postgres_repo else PostgresREPO()
        self.milvus_repo = milvus_repo if milvus_repo else MilvusREPO()
        self.vec_handler = vec_handler if vec_handler else VectorHandler()
    
    def init_data(self) -> None:
        self.postgres_repo.create_table()

        self.postgres_repo.transfer_data_from_csv_to_postgres(CSV_PATH)

        # self.sync_postgres_milvus_in_bulk()
        self.milvus_repo.rebuild_collection() # ~~ko thể dùng rebuild để init~~
                                              # có thể dùng rebuild để init

    # def sync_postgres_milvus_in_bulk(self) -> None: # vậy thì cần gì hàm này nữa, gọi trực tiếp rebuild collection cho r 
    #     # corpus = self.postgres_repo.get_corpus()
    #     # vectors, unique_words = self.vec_handler.generate_vectors(corpus)

    #     # self.milvus_repo.create_collection(len(unique_words))

    #     # movie_data = self.postgres_repo.get_all_movies()
    #     # self.milvus_repo.store_vectors_to_milvus(vectors, movie_data)
    #     self.milvus_repo.rebuild_collection() # rebuild đã bao gồm tất cả những thứ trên và hoàn thiện hơn
    # 
    
    

if __name__ == "__main__":
    dm = DataManager()
    dm.init_data()