import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from repositories.concrete.postgres_repo import PostgresREPO
from repositories.concrete.milvus_repo import MilvusREPO
from psycopg2.extras import RealDictCursor
import dependencies.tfidf as tfidf

pg = PostgresREPO()

#cần xem lại các func nên thuộc về ai
#utilities là dependency?

class utilities:
    def movie_dataset_processing_from_postgres(self):
        conn = pg.connect_to_postgres()
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

    def get_unique_words_for_vector_dim(self, corpus: list[str])->list[str]:
        unique_words = set()
        for doc in corpus:
            vocab = tfidf.create_vocab_single(doc)
            unique_words.update(vocab.keys())  
        return sorted(list(unique_words))
            #nên thuộc tfidf?


    def generate_tfidf(self):
        dataset = self.movie_dataset_processing_from_postgres()
        vocab_all = tfidf.create_vocab_all(dataset)
        tf_all = tfidf.compute_tf_all(vocab_all)
        idf_dict = tfidf.compute_idf_single(dataset)
        tfidf_all = tfidf.compute_tfidf_all(tf_all, idf_dict)
        return dataset, tfidf_all, idf_dict
            #nên thuộc tfidf?

    
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

    def invalidate_cache(self):
        self.idf_dict = None
        self.unique_words = None
        self._corpus_cache = None
        #ko dùng self được

    def update_dict_after_crud(self):
        self.invalidate_cache()
        corpus = self.movie_dataset_processing_from_postgres()
        _, _, self.idf_dict = self.generate_tfidf()
        self.unique_words = self.get_unique_words_for_vector_dim(corpus)
        #ko dùng self được
        #nên thuộc tfidf?
