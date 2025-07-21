import os
import pandas as pd
import tfidf 
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "movies.csv")
VECTOR_FILE = os.path.join(BASE_DIR, "vector_db.pkl")

def movie_dataset_processing(source='ml-latest-small/movies.csv'):
    movie = pd.read_csv(source) 
    movie['text'] = movie['title'] + ' | ' + movie['genres'] 
    dataset = movie['text'].tolist() 
    return dataset

def generate_vectors():
    dataset = movie_dataset_processing(DATASET_PATH)
    dataset_vocab = tfidf.create_vocab_all(dataset)
    tf_all = tfidf.compute_tf_all(dataset_vocab)
    idf_dict = tfidf.compute_idf_single(dataset)
    tfidf_all = tfidf.compute_tfidf_all(tf_all, idf_dict)
    vectorized_dataset = tfidf_all
    return dataset, vectorized_dataset

def storing_vectors(dataset: list[str], vectorized_dataset: list[dict]):    
    vector_store = []
    for text, vector in zip(dataset, vectorized_dataset):
        if not isinstance(vector, dict):
            raise ValueError(f"Vector at text '{text[:30]}...' is not a dict.")
        vector_store.append({"text": text, "vector": vector})
    with open(VECTOR_FILE, "wb") as f:
        pickle.dump(vector_store, f)

def loading_vectors() -> list[dict]:
    if not os.path.exists(VECTOR_FILE):
        raise FileNotFoundError(f"{VECTOR_FILE} does not exist.")
    with open(VECTOR_FILE, "rb") as f:
        vector_store = pickle.load(f)
    return vector_store

def initialize_database():
    if not os.path.exists(VECTOR_FILE):
        print("Vector database not found. Creating new one...")
        dataset, vectorized_datatset = generate_vectors()
        storing_vectors(dataset, vectorized_datatset)
        print("Vector database created successfully!")
    else:
        print("Vector database already exists.")

if __name__ == "__main__":
    initialize_database()