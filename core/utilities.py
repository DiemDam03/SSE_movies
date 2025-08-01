import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import core.tfidf as tfidf

class VectorHandler:
    def get_unique_words(self, corpus: list[str])->list[str]:
        unique_words = set()
        for doc in corpus:
            vocab = tfidf.create_vocab_single(doc)
            unique_words.update(vocab.keys())  
        return sorted(list(unique_words))

    def generate_tfidf(self, corpus: list[str]) -> tuple[list[dict], dict]:
        vocab_all = tfidf.create_vocab_all(corpus)
        tf_all = tfidf.compute_tf_all(vocab_all)
        idf_dict = tfidf.compute_idf_single(corpus)
        tfidf_all = tfidf.compute_tfidf_all(tf_all, idf_dict)
        return tfidf_all, idf_dict

    def converting_tfidf_to_fixed_dim_vector(self, tfidf_dict: dict, unique_words: list[str])->list[float]:
        vector = [0.0] * len(unique_words)
        word_to_index = {word: i for i, word in enumerate(unique_words)}
        for word, value in tfidf_dict.items():
            if word in word_to_index:
                vector[word_to_index[word]] = value
        return vector

    def generate_vectors(self, corpus: list[str]) -> tuple[list[list[float]], list[str]]:
        tfidf_all, _ = self.generate_tfidf(corpus)
        unique_words = self.get_unique_words(corpus)
        modified_vectors = [
            self.converting_tfidf_to_fixed_dim_vector(tfidf_dict, unique_words)
            for tfidf_dict in tfidf_all
        ]
        return modified_vectors, unique_words
    
