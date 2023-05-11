import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

def get_data():
    df = pd.read_csv('news_cleaned_sentiment.csv')
    # converting content to text
    df['content'] = df['content'].astype(str)
    return df


def preprocess_text(df):
    def preprocess(token):
        tokens = simple_preprocess(token, deacc=True)  # Tokenize and remove punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
        return tokens
    
    # Preprocess the content for each article in the DataFrame
    df['preprocessed_content'] = df['content'].apply(preprocess)
    return df


def train_word2vec_model(df, model_name):
    # Assuming 'preprocessed_content' is the column containing preprocessed text data
    sentences = df['preprocessed_content'].tolist()
    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    # Save the model
    model.save(model_name)


def save_model(model, source):
    model.save(f"{source}_word2vec_model")


if __name__ == "__main__":
    df = get_data()
    df = preprocess_text(df)

    news_sources = ['CNN', 'MSNBC', 'Fox News']
    word2vec_models = {}

    for source in news_sources:
        source_articles = df[df['source_name'] == source]['preprocessed_content'].tolist()
        model = train_word2vec_model(source_articles)
        word2vec_models[source] = model

    for source, model in word2vec_models.items():
        save_model(model, source)
