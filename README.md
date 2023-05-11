# NLP Word Embedding Analysis

This project uses NLP (Natural Language Processing) and Word2Vec embeddings to perform comparative analysis of news articles from different sources.

## Overview

The data used in this project was fetched and cleaned in a previous project. You can find the details about that process [here](link_to_previous_project).

The main focus of this project is to train Word2Vec models on text data from various news sources, and use these models to analyze and visualize the semantic relationships between words.

## About Word Embeddings

Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation. They are a distributed representation for text that is perhaps one of the key breakthroughs for the impressive performance of deep learning methods on challenging natural language processing problems.

Word2Vec is a popular algorithm used for generating dense vector representations of words in large corpora using unsupervised learning. The resulting vectors have been found to exhibit interesting semantic properties, such as clustering of similar words and the ability to answer analogy questions, e.g., "man is to woman as king is to queen".

## Project Structure

Here's a brief overview of the Python scripts in this project:

1. `train_embedding.py`: This script contains functions to load and preprocess the data, and train Word2Vec models.

2. `reduce_n_plot.py`: This script contains functions to reduce the dimensionality of the word embeddings using t-SNE, and visualize them using matplotlib.

3. `main.py`: This is the main script that ties everything together. It loads and preprocesses the data, trains the Word2Vec models, loads the models, retrieves and plots similar words and word analogies.

## Getting Started

1. Clone the repository: `git clone https://github.com/yourusername/nlp_word_embedding_analysis.git`

2. Install the requirements: `pip install -r requirements.txt`

3. Run the main script: `python main.py`

## Further Work

This is just the beginning of what can be done with word embeddings and NLP. Potential extensions of this project could involve comparing more news sources, experimenting with different word embedding algorithms, or using the word embeddings as input to a machine learning model to perform tasks such as text classification or sentiment analysis.
