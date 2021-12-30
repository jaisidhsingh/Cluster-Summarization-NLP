import math
import nltk
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import sklearn
from transformers import BartTokenizer, BartForConditionalGeneration
import transformers
import torch

nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import sent_tokenize, word_tokenize

warnings.filterwarnings("ignore")