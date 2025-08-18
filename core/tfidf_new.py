import re
import numpy as np
class VectorLord:
    def __init__(self):
        self.idf_dict = {}
        self.vocab = {}

    def create_tokens_single(self, doc: str) -> tuple:
        tokens = doc.lower()
        tokens = re.sub(r"[^\w\s]", "", tokens)
        tokens = re.sub(r"\s+", " ", tokens).strip()
        return tuple(tokens.split())

    def create_vocab_single(self, doc: str) -> dict[str, int]:
        all_tokens = self.create_tokens_single(doc)
        word_dict = {}
        for word in all_tokens:
            word_dict[word] = word_dict.get(word, 0) + 1
        return word_dict

    def create_vocab_all(self, corpus: list[str]) -> None:
        vocab_set = set()
        for doc in corpus:
            word_dict = self.create_vocab_single(doc)
            vocab_set.update(word_dict.keys())
        
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab_set))}
    
    def compute_tf(self, word_dict: dict[str, int]) -> np.ndarray:
        total_terms = sum(word_dict.values())
        tf_vector = np.zeros(len(self.vocab))

        for word, count in word_dict.items():
            if word in self.vocab:
                idx = self.vocab[word]
                tf_vector[idx] = count / total_terms
        return tf_vector
    
    def compute_idf(self, corpus: list[str]) -> np.ndarray:

        total_docs = len(corpus)
        doc_freq = np.zeros(len(self.vocab))

        for doc in corpus:
            word_set = set(self.create_vocab_single(doc).keys())
            for word in word_set:
                if word in self.vocab:
                    idx = self.vocab[word]
                    doc_freq[idx] +=1 

        idf_vector = np.log(1 + total_docs / (doc_freq + 1e-8))
        self.idf_dict = idf_vector
    
        return idf_vector

    def compute_tfidf(self, corpus: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not self.vocab:
            self.create_vocab_all()

        idf = self.compute_idf(corpus)

        tfidf = np.zeros(len(corpus), len(self.vocab))

        for i, doc in enumerate(corpus):
            word_dict = self.create_vocab_single(doc)
            tf_vector = self.compute_tf(word_dict)
            tfidf[i] = tf_vector * idf
        
        return tfidf, idf





    # def compute_tf_single(self, word_dict: dict) -> dict:
    #     tf_dict = {}
    #     total_terms = sum(word_dict.values())
    #     for word, count in word_dict.items():
    #         tf_dict[word] = count / total_terms
    #     return tf_dict

    # def compute_tf_all(self, vocab_all: list[dict]) -> list[dict]:
    #     tf_list = []
    #     for word_dict in vocab_all:
    #         tf_list.append(self.compute_tf_single(word_dict))
    #     return tf_list

    # def compute_idf(self, corpus: list[str]) -> dict:
    #     idf_dict = {}
    #     total_docs = len(corpus)
    #     for doc in corpus:
    #         word_set = set(self.create_vocab_single(doc).keys())
    #         for word in word_set:
    #             idf_dict[word] = idf_dict.get(word, 0) + 1
    #     for word, doc_count in idf_dict.items():
    #         idf_dict[word] = math.log((total_docs / (doc_count + 1)) + 1) 
    #     return idf_dict

    # def compute_tfidf_all(self, tf_list: list[dict], idf_dict: dict) -> list[dict]:
    #     tfidf_list = []
    #     for tf_dict in tf_list:
    #         tfidf_dict = {}
    #         for word, tf in tf_dict.items():
    #             idf = idf_dict.get(word, 0.0)
    #             tfidf_dict[word] = tf * idf
    #         tfidf_list.append(tfidf_dict)
    #     return tfidf_list

    # def compute_tfidf_single(self, tf_dict: dict, idf_dict: dict) -> dict:
    #     return {word: tf_dict[word] * idf_dict.get(word, 0.0) for word in tf_dict}

    # def get_unique_words(self, corpus: list[str])->list[str]:
    #     unique_words = set()
    #     for doc in corpus:
    #         vocab = self.create_vocab_single(doc)
    #         unique_words.update(vocab.keys())  
    #     return sorted(list(unique_words))

    # def generate_tfidf(self, corpus: list[str]) -> tuple[list[dict], dict]:
    #     vocab_all = self.create_vocab_all(corpus)
    #     tf_all = self.compute_tf_all(vocab_all)
    #     idf_dict = self.compute_idf(corpus)
    #     tfidf_all = self.compute_tfidf_all(tf_all, idf_dict)
    #     return tfidf_all, idf_dict

    # def converting_tfidf_to_fixed_dim_vector(self, tfidf_dict: dict, unique_words: list[str])->list[float]:
    #     vector = [0.0] * len(unique_words)
    #     word_to_index = {word: i for i, word in enumerate(unique_words)}
    #     for word, value in tfidf_dict.items():
    #         if word in word_to_index:
    #             vector[word_to_index[word]] = value
    #     return vector

    # def generate_vectors(self, corpus: list[str]) -> tuple[list[list[float]], list[str]]:
    #     tfidf_all, _ = self.generate_tfidf(corpus)
    #     unique_words = self.get_unique_words(corpus)
    #     modified_vectors = [
    #         self.converting_tfidf_to_fixed_dim_vector(tfidf_dict, unique_words)
    #         for tfidf_dict in tfidf_all
    #     ]
    #     return modified_vectors, unique_words