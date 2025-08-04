import re
import math

def create_tokens_single(doc: str) -> list[str]:
  tokens = doc.lower()
  tokens = re.sub(r"[^\w\s]", "", tokens)
  tokens = re.sub(r"\s+", " ", tokens).strip()
  return tokens.split()

def create_vocab_single(doc: str) -> dict:
  all_tokens = create_tokens_single(doc)
  word_dict = {}
  for word in all_tokens:
      word_dict[word] = word_dict.get(word, 0) + 1
  return word_dict

def create_vocab_all(corpus: list[str]) -> list[dict]:
  vocab_dicts = []
  for doc in corpus:
      word_dict = create_vocab_single(doc)
      vocab_dicts.append(word_dict)
  return vocab_dicts

def compute_tf_single(word_dict: dict) -> dict:
    tf_dict = {}
    total_terms = sum(word_dict.values())
    for word, count in word_dict.items():
        tf_dict[word] = count / total_terms
    return tf_dict

def compute_tf_all(vocab_all: list[dict]) -> list[dict]:
    tf_list = []
    for word_dict in vocab_all:
        tf_list.append(compute_tf_single(word_dict))
    return tf_list

def compute_idf_single(corpus: list[str]) -> dict:
    idf_dict = {}
    total_docs = len(corpus)
    for doc in corpus:
        word_set = set(create_vocab_single(doc).keys())
        for word in word_set:
            idf_dict[word] = idf_dict.get(word, 0) + 1
    for word, doc_count in idf_dict.items():
        idf_dict[word] = math.log(1 + total_docs / doc_count)
    return idf_dict

def compute_tfidf_all(tf_list: list[dict], idf_dict: dict) -> list[dict]:
    tfidf_list = []
    for tf_dict in tf_list:
        tfidf_dict = {}
        for word, tf in tf_dict.items():
            idf = idf_dict.get(word, 0.0)
            tfidf_dict[word] = tf * idf
        tfidf_list.append(tfidf_dict)
    return tfidf_list

def compute_tfidf_single(tf_dict: dict, idf_dict: dict) -> dict:
    return {word: tf_dict[word] * idf_dict.get(word, 0.0) for word in tf_dict}
