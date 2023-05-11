from load_and_clean_data import load_and_clean_data
from train_word2vec_model import train_word2vec_model
from reduce_n_plot import load_models, get_similar_words, reduce_dimensions, plot_reduced_embeddings, plot_word_analogy_2d

def main():
    # Load and clean data
    cnn_data, msnbc_data, fox_data = load_and_clean_data()

    # Train Word2Vec models
    train_word2vec_model(cnn_data, "CNN_word2vec_model")
    train_word2vec_model(msnbc_data, "MSNBC_word2vec_model")
    train_word2vec_model(fox_data, "Fox News_word2vec_model")

    # Load the models
    cnn_model, msnbc_model, fox_model = load_models()

    # Get similar words and their embeddings
    similar_words, embeddings = get_similar_words([cnn_model, msnbc_model, fox_model], 'economy', 10)

    # Reduce dimensions
    embeddings_2d = reduce_dimensions(embeddings)

    # Plot reduced embeddings
    plot_reduced_embeddings(embeddings_2d, similar_words, 10, 'economy')

    # Plot word analogy
    plot_word_analogy_2d(cnn_model, 'washington', 'republicans', 'trump', 'liberal')

if __name__ == "__main__":
    main()
