from utils.sentence_embedding import *
from utils.clustering import *
from models.summarizers import *
from nltk.tokenize import sent_tokenize, word_tokenize
import math
from time import perf_counter
import time


def get_summary(model_name, article, max_length, min_length, increment):
	start_time = perf_counter()
	summarization_model, summarization_tokenizer = load_summarizer(model_name)
	summarizer_token_limit = summarization_tokenizer.model_max_length
	print("Going Beyong Token limit:", summarizer_token_limit)

	input_word_toks = word_tokenize(article)
	num_words = len(input_word_toks)

	if num_words <= summarizer_token_limit and model_name == "t5":
		pred_summary = summarize_input(article, summarization_model, summarization_tokenizer)		
		end_time = perf_counter()
		print("Time taken: ", end_time - start_time)
	
	else:
		input_sent_toks = sent_tokenize(article)
		embeddings = make_embeddings(input_sent_toks, mean_pooling)
		embeddings = embeddings.numpy()

		increment[0] = 20

		n_clusters_estimate = math.ceil(num_words / summarizer_token_limit)

		clemb = ClusterEmbeddings(
			cluster_estimate=n_clusters_estimate,
			cluster_fn="agglo", # much better
			embeddings=embeddings,
			sentences=np.array(input_sent_toks),
			words=np.array(input_word_toks)
		)

		increment[0] = 50

		sentence_clusters = clemb.get_sentence_clusters()	

		n = len(sentence_clusters)
		summs = ""
		for cluster in sentence_clusters:
			cluster_summary = summarize_input(
				cluster, 
				summarization_model, 
				summarization_tokenizer,
				max_length=250,
				min_length=50,
			)
			if type(cluster_summary) == list:
				cluster_summary = cluster_summary[0]
			summs += cluster_summary + " "
			
			increment[0] += 40 / n

		pred_summary = summarize_input(
			summs, 
			summarization_model, 
			summarization_tokenizer,
			max_length=max_length,
			min_length=min_length,
		)

		increment[0] += 100

		end_time = perf_counter()
		time_taken = end_time - start_time
	
	return pred_summary, time_taken

def test():
	article = """Recent text-to-image matching models apply contrastive learning to large corpora of uncurated pairs of images and sentences. While such models can provide a powerful score for matching and subsequent zero-shot tasks, they are not capable of generating caption given an image. In this work, we repurpose such models to generate a descriptive text given an image at inference time, without any further training or tuning step. This is done by combining the visual-semantic model with a large language model, benefiting from the knowledge in both web-scale models. The resulting captions are much less restrictive than those obtained by supervised captioning methods. Moreover, as a zero-shot learning method, it is extremely flexible and wedemonstrate its ability to perform image arithmetic in which the inputs can be either images or text and the output is a sentence."""
	model_name = "BART"
	summ, time_taken = get_summary(model_name, article, 250, 150)
	print(summ)
	print(time_taken)

