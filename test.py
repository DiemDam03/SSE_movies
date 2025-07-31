from db import DatabaseManager
import os
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "movies.csv")
dbm = DatabaseManager()

def movie_dataset_processing(source = CSV_PATH):
    movie = pd.read_csv(source) # Load the movie dataset
    movie['text'] = movie['title'] + ' | ' + movie['genres'] # Combine title and genres into a single text column
    corpus = movie['text'].tolist() 
    return corpus
