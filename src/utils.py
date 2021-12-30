import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings("ignore")


def make_data(articles):
    sentences = []
    words = []

    for item in articles:
        tokenized_article = sent_tokenize(item)
        for sentence in tokenized_article:
            word_tokens = word_tokenize(sentence)
            for token in word_tokens:
                words.append(token)
            sentences.append(sentence)
    sentences, words = np.array(sentences), np.array(words)
    return sentences, words


def sentence2embedding(sentences_array, model):
    return model.encode(sentences_array)


def bart_summarize(
    text,
    model,
    tokenizer,
    device='cpu',
    num_beams=3,
    length_penalty=2.0,
    max_length=200,
    min_length=50,
    no_repeat_ngram_size=3,
):

    text = text.replace('\n', '')
    text_input_ids = tokenizer.batch_encode_plus(
        [text], return_tensors='pt', max_length=1024)['input_ids'].to(device)
    summary_ids = model.generate(text_input_ids, num_beams=int(num_beams), length_penalty=float(
        length_penalty), max_length=int(max_length), min_length=int(min_length), no_repeat_ngram_size=int(no_repeat_ngram_size))
    summary_txt = tokenizer.decode(
        summary_ids.squeeze(), skip_special_tokens=True)
    return summary_txt


class ClusterSentenceEmbeddings():
    def __init__(self, clustering_algorithm, embeddings, sentences, words, num_articles):
        self.num_articles = num_articles
        self.embeddings = embeddings
        self.sentences = sentences
        self.words = words
        self.clustering = clustering_algorithm.fit(self.embeddings)

    def cluster_outputs(self):
        outputs = {
            'num_clusters': self.clustering.n_clusters_,
            'labels': self.clustering.labels_,
            'components': self.clustering.n_connected_components_,
            'distances': self.clustering.distances_,
            'exp_sentences_per_cluster': math.ceil(
                len(self.sentences)/self.clustering.n_clusters_
            ),  # using pigeonhole principle formula for expectated number of pigeons per hole.
            'exp_words_per_cluster': math.ceil(
                len(self.words)/self.clustering.n_clusters_
            ),  # using pigeonhole principle formula for expected nu,ber of pigeons per hole.
        }
        return outputs

    def visualize_outputs(self, method='tsne', dims=2):

        if method == 'tsne':
            tSNE = TSNE(n_components=dims, learning_rate='auto', init='random')
            projected_embeddings = np.array(
                tSNE.fit_transform(self.embeddings))

            cluster_outputs = self.cluster_outputs()

            indiv_clusters = []
            for label in range(cluster_outputs['num_clusters']):
                indiv_clusters.append(projected_embeddings[
                    cluster_outputs['labels'] == label
                ])

            plt.figure(figsize=(15, 15))
            for i, cluster in enumerate(indiv_clusters):
                scatter = plt.scatter(
                    cluster[:, :1],
                    cluster[:, 1:],
                    color=plt.cm.get_cmap(
                        'hsv', cluster_outputs['num_clusters']+1)(i),
                    alpha=0.6,
                    label=i,
                    s=200,
                )
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('tSNE Results: Sentence Embeddings')
            plt.legend(title='Clustering Labels')
            plt.show()
            print('\n')

        if method == 'pca':
            pca = PCA(n_components=dims)
            projected_embeddings = np.array(pca.fit_transform(self.embeddings))

            cluster_outputs = self.cluster_outputs()

            indiv_clusters = []
            for label in range(cluster_outputs['num_clusters']):
                indiv_clusters.append(projected_embeddings[
                    cluster_outputs['labels'] == label
                ])

            plt.figure(figsize=(15, 15))
            for i, cluster in enumerate(indiv_clusters):
                scatter = plt.scatter(
                    cluster[:, :1],
                    cluster[:, 1:],
                    color=plt.cm.get_cmap(
                        'hsv', cluster_outputs['num_clusters']+1)(i),
                    alpha=0.6,
                    label=i,
                    s=200,
                )
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('PCA Results: Sentence Embeddings')
            plt.legend(title='Clustering Labels')
            plt.show()
            print('\n')

        return None

    def get_sentence_clusters(self):
        cluster_outputs = self_cluster_outputs()
        indiv_clusters = []
        chunk = ""

        for label in range(cluster_outputs['num_clusters']):
            single_cluster = self.sentencesp[cluster_outputs['labels'] == label]
            for item in single_cluster:
                chunk += item + " "
            indiv_clusters.append(chunk)
            chunk = ""

        return np.array(indiv_clusters)
