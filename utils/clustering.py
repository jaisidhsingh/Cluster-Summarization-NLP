import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


class ClusterEmbeddings():
	def __init__(
		self,
		cluster_estimate,
		cluster_fn,
		embeddings,
		sentences,
		words
	):
		self.cluster_estimate = cluster_estimate
		self.embeddings = embeddings
		self.sentences = sentences
		self.words = words
		
		self.cluster_fn = cluster_fn
		if self.cluster_fn == "agglo":
			self.clustering_algo = AgglomerativeClustering(n_clusters=self.cluster_estimate)
			self.num_clusters = cluster_estimate

		elif self.cluster_fn == "kmeans":
			self.clustering_algo = KMeans(n_clusters=self.cluster_estimate)
			self.num_clusters = cluster_estimate

		self.cluster = self.clustering_algo.fit(embeddings)
		self.labels = self.cluster.labels_

	def get_sentence_clusters(self):
		sent_clusters = []
		chunk = ""

		for lbl in range(self.num_clusters):
			single_cluster = self.sentences[self.labels == lbl]
			for sent in single_cluster:
				chunk += sent + " "
			sent_clusters.append(chunk)
			chunk = ""
		
		return np.array(sent_clusters)

	def make_plot(self):
		projector = TSNE(
			n_components=2,
			learning_rate="auto",
			init="random"
		)
		proj_embeddings = np.array(
			projector.fit_transform(self.embeddings)
		)

		for lbl in range(self.num_clusters):
			xs = proj_embeddings[self.labels == lbl]
			plt.scatter(xs[:, 0], xs[:, 1], label=f"Cluster {lbl}")

		plt.legend()
		plt.xlabel("x1")
		plt.ylabel("x2")
		plt.show()