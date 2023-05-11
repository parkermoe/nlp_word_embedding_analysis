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


def train_word2vec_model(source_articles):
    model = Word2Vec(source_articles, size=100, window=5, min_count=5, workers=4)
    model.train(source_articles, total_examples=len(source_articles), epochs=10)
    return model


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
