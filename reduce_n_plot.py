from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import sklearn

def load_models():
    cnn_model = Word2Vec.load('CNN_word2vec_model')
    msnbc_model = Word2Vec.load('MSNBC_word2vec_model')
    fox_model = Word2Vec.load('Fox News_word2vec_model')
    return cnn_model, msnbc_model, fox_model

def get_similar_words(models, word, num_similar_words):
    similar_words = [word]
    embeddings = []
    for model in models:
        model_similar_words = [word for word, _ in model.wv.most_similar(word, topn=num_similar_words)]
        similar_words += model_similar_words
        embeddings += [model.wv.get_vector(w) for w in model_similar_words]
    embeddings = [models[0].wv.get_vector(word)] + embeddings
    embeddings = np.array(embeddings)
    return similar_words, embeddings

def reduce_dimensions(embeddings):
    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

def plot_reduced_embeddings(embeddings_2d, similar_words, num_similar_words, word):
    fig = go.Figure()
    def add_traces(fig, start, end, name, color):
        # Helper function to add traces with alternating label positions
        for i in range(start, end):
            if i == start:
                fig.add_trace(go.Scatter(x=[embeddings_2d[i, 0]], 
                                         y=[embeddings_2d[i, 1]],
                                         mode='markers+text',
                                         name=name,
                                         text=[similar_words[i]],
                                         textposition='bottom center' if i%2 == 0 else 'top center',
                                         marker=dict(color=color)))
            else:
                fig.add_trace(go.Scatter(x=[embeddings_2d[i, 0]], 
                                         y=[embeddings_2d[i, 1]],
                                         mode='markers+text',
                                         name="",
                                         text=[similar_words[i]],
                                         textposition='bottom center' if i%2 == 0 else 'top center',
                                         marker=dict(color=color),
                                         showlegend=False))

    add_traces(fig, 1, num_similar_words+1, 'CNN', 'red')
    add_traces(fig, num_similar_words+1, 2*num_similar_words+1, 'MSNBC', 'blue')
    add_traces(fig, 2*num_similar_words+1, len(embeddings_2d), 'Fox News', 'green')
    fig.add_trace(go.Scatter(x=[embeddings_2d[0, 0]], 
                             y=[embeddings_2d[0, 1]],
                             mode='markers+text',
                             name='Original Word',
                             text=[word],
                             textposition='top center',
                             marker=dict(color='black')))

    fig.update_layout(
        title=f"Top 10 Most Similar Words to '{word}' for CNN, MSNBC, and Fox News",
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        legend_title="News Source",
        font=dict(family="Courier New, monospace", size=8),
        hovermode='closest',
        width=500,
        height=500,
        autosize=False,
        margin=dict(l=50, r=50, b=50, t=50, pad=10),
    )

    fig.show()

def plot_word_analogy_2d(model, word1, word2, word3, word4):
        # Get the word embeddings
        word1_vec = model.wv[word1]
        word2_vec = model.wv[word2]
        word3_vec = model.wv[word3]
        word4_vec = model.wv[word4]

        words_vec = np.array([word1_vec, word2_vec, word3_vec, word4_vec])

        # Reduce dimensions
        tsne = TSNE(n_components=2, random_state=42, perplexity=2)
        words_vec_2d = tsne.fit_transform(words_vec)

        plt.figure(figsize=(8, 8))

        # Plot the points
        plt.scatter(words_vec_2d[:, 0], words_vec_2d[:, 1], color=['red', 'blue', 'green', 'purple'])

        # Annotate the points
        plt.text(words_vec_2d[0, 0], words_vec_2d[0, 1], word1, ha='right')
        plt.text(words_vec_2d[1, 0], words_vec_2d[1, 1], word2, ha='right')
        plt.text(words_vec_2d[2, 0], words_vec_2d[2, 1], word3, ha='right')
        plt.text(words_vec_2d[3, 0], words_vec_2d[3, 1], word4, ha='right')

        # Draw the arrows
        plt.arrow(words_vec_2d[0, 0], words_vec_2d[0, 1], words_vec_2d[1, 0] - words_vec_2d[0, 0], words_vec_2d[1, 1] - words_vec_2d[0, 1], shape='full', lw=1, length_includes_head=True, head_width=.02, color='red')
        plt.arrow(words_vec_2d[2, 0], words_vec_2d[2, 1], words_vec_2d[3, 0] - words_vec_2d[2, 0], words_vec_2d[3, 1] - words_vec_2d[2, 1], shape='full', lw=1, length_includes_head=True, head_width=.02, color='blue')

        plt.title('Word 1 : Word 2 :: Word 3 : Word 4')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True)
        plt.show()

