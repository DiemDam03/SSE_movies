import re
import math
from typing import List, Dict, Tuple


def create_tokens_single(doc: str) -> List[str]:
    tokens = doc.lower()
    tokens = re.sub(r"[^\w\s]", "", tokens)
    tokens = re.sub(r"\s+", " ", tokens).strip()
    return tokens.split()

def create_vocab_single(doc: str) -> Dict[str, int]:
    all_tokens = create_tokens_single(doc)
    word_dict = {}
    for word in all_tokens:
        word_dict[word] = word_dict.get(word, 0) + 1
    return word_dict

def create_vocab_all(corpus: List[str]) -> List[Dict[str, int]]:
    vocab_dicts = []
    for doc in corpus:
        word_dict = create_vocab_single(doc)
        vocab_dicts.append(word_dict)
    return vocab_dicts

def compute_tf_single(word_dict: Dict[str, int]) -> Dict[str, float]:
    tf_dict = {}
    total_terms = sum(word_dict.values())
    if total_terms == 0:
        return tf_dict
    for word, count in word_dict.items():
        tf_dict[word] = count / total_terms
    return tf_dict

def compute_tf_all(vocab_all: List[Dict[str, int]]) -> List[Dict[str, float]]:
    tf_list = []
    for word_dict in vocab_all:
        tf_list.append(compute_tf_single(word_dict))
    return tf_list

def compute_idf_single(corpus: List[str]) -> Dict[str, float]:
    idf_dict = {}
    total_docs = len(corpus)
    if total_docs == 0:
        return idf_dict
        
    for doc in corpus:
        word_set = set(create_vocab_single(doc).keys())
        for word in word_set:
            idf_dict[word] = idf_dict.get(word, 0) + 1
    
    for word, doc_count in idf_dict.items():
        idf_dict[word] = math.log(1 + total_docs / doc_count)
    return idf_dict

def compute_tfidf_all(tf_list: List[Dict[str, float]], idf_dict: Dict[str, float]) -> List[Dict[str, float]]:
    tfidf_list = []
    for tf_dict in tf_list:
        tfidf_dict = {}
        for word, tf in tf_dict.items():
            idf = idf_dict.get(word, 0.0)
            tfidf_dict[word] = tf * idf
        tfidf_list.append(tfidf_dict)
    return tfidf_list

def compute_tfidf_single(tf_dict: Dict[str, float], idf_dict: Dict[str, float]) -> Dict[str, float]:
    return {word: tf_dict[word] * idf_dict.get(word, 0.0) for word in tf_dict}

def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    common_words = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[word] * vec2[word] for word in common_words)
    norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
    norm2 = math.sqrt(sum(val**2 for val in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def ranking(query_tfidf: Dict[str, float], tfidf_list: List[Dict[str, float]], top_k: int) -> List[Tuple[int, float]]:
    scores = []
    for i, doc_tfidf in enumerate(tfidf_list):
        score = cosine_similarity(query_tfidf, doc_tfidf)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]