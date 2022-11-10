from .utils import *
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import argparse
import warnings
from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("ignore")

def get_summary(articles, threshold):
	EMBEDDING_MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')
	SUMMARIZATION_TOKENIZER = BartTokenizer.from_pretrained(
	    'facebook/bart-large-cnn')
	SUMMARIZATION_MODEL = BartForConditionalGeneration.from_pretrained(
	    'facebook/bart-large-cnn')
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
