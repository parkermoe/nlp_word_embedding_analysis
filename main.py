from train_embedding import get_data, preprocess_text
from train_embedding import train_word2vec_model
from reduce_n_plot import load_models, get_similar_words, reduce_dimensions, plot_reduced_embeddings, plot_word_analogy_2d

def main():
    # Load and clean data
    df = get_data()
    df = preprocess_text(df)

    news_sources = ['CNN', 'MSNBC', 'Fox News']
    for source in news_sources:
        source_data = df[df['source_name'] == source]
        # Train Word2Vec models
        train_word2vec_model(source_data, f"{source}_word2vec_model")

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
