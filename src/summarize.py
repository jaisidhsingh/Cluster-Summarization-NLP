from models import *
from utils import *
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")


def get_summary(articles, threshold):
	sentences, words = make_data(articles)
	embeddings = sentence2embedding(sentences, EMBEDDING_MODEL)

	CUSTOM_CLEMB_KWARGS = {
		'clustering_algorithm': AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=threshold
		),
		'embeddings': embeddings,
		'sentences': sentences,
		'words': words,
		'num_articles': len(articles)
	}

	clemb = ClusterSentenceEmbeddings(**CUSTOM_CLEMB_KWARGS)
	sentence_clusters = clemb.get_sentence_clusters()

	final_summary = ""
	for cluster in sentence_clusters:
		summary = bart_summarize(cluster, SUMMARIZATION_MODEL, SUMMARIZATION_TOKENIZER)
		final_summary += summary + " "

	return final_summary
