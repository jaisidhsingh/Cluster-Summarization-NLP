from load_examples import example_set
from utils import *
from models import *
from sklearn.cluster import AgglomerativeClustering

articles = example_set()
sentences, words = make_data(articles)
embeddings = sentence2embedding(sentences, EMBEDDING_MODEL)

CLEMB_KWARGS = {
    'clustering_algorithm': AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=7
    ),
    'embeddings': embeddings,
    'sentences': sentences,
    'words': words,
    'num_articles': len(articles)
}
