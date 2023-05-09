import streamlit as st
from summarize import *
from utils.sentence_embedding import *
from utils.clustering import *
from models.summarizers import *
from nltk.tokenize import sent_tokenize, word_tokenize
import math
from time import perf_counter


START = False
COMPLETED = False
PLACEHOLDER = "Enter your article"

st.markdown("Extractive Summarization for Large Articles 😊")

article = st.text_input(
    label="Welcome, enter your article, press enter, and then Summarize",
    value=PLACEHOLDER,
)

model_name = st.sidebar.selectbox(
    label="Pick your model of choice:",
    options=("BART", "Pegasus", "Distill-BART", "RoBERTa")
)

max_length = st.sidebar.slider(
	label="Choose the maximum length of the summary",
	min_value=100,
	max_value=500,
	value=250
)

min_length = st.sidebar.slider(
	label="Choose the minimum length of the summary",
	min_value=20,
	max_value=150,
	value=50
)

go = st.button(
    label="Summarize",
    key=0,
)

reset = st.button(
	label="Reset",
	key=1,
)


START = go
tmp_out = st.empty()

if reset:
	COMPLETED = not reset
	tmp_out.empty()
else:
	COMPLETED = reset


bar = st.progress(0)

if START and not COMPLETED:
	start_time = perf_counter()
	
	with tmp_out.container():
		st.write("Loading in models and preparing article...")
	
	summarization_model, summarization_tokenizer = load_summarizer(model_name)
	summarizer_token_limit = summarization_tokenizer.model_max_length

	if "pegasus" in model_name.lower():
		input_toks = sent_tokenize(article)
		input_sent_toks = input_toks
		input_word_toks = word_tokenize(article)
		num_toks = len(input_toks)
	else:
		input_toks = word_tokenize(article)
		input_word_toks = input_toks
		input_sent_toks = sent_tokenize(article)
		num_toks = len(input_toks)

	bar.progress(15)

	if num_toks <= summarizer_token_limit:
		with tmp_out.container():
			st.write("Input token count (",num_toks,") <= token limit (",summarizer_token_limit,"), skipping optimization ...")
		
		pred_summary = summarize_input(article, summarization_model, summarization_tokenizer)		
		end_time = perf_counter()
		time_taken = end_time - start_time
		bar.progress(100)

	else:
		with tmp_out.container():
			st.write("Input token count (",num_toks,") > token limit (",summarizer_token_limit,"), optimizing ...")
			st.write(f"Going Beyong {model_name} Token limit:", summarizer_token_limit)

		input_sent_toks = sent_tokenize(article)
		embeddings = make_embeddings(input_sent_toks, mean_pooling)
		embeddings = embeddings.numpy()

		bar.progress(30)

		n_clusters_estimate = math.ceil(num_toks / summarizer_token_limit)

		clemb = ClusterEmbeddings(
			cluster_estimate=n_clusters_estimate,
			cluster_fn="agglo", # much better
			embeddings=embeddings,
			sentences=np.array(input_sent_toks),
			words=np.array(input_word_toks)
		)

		bar.progress(50)
		curr = 50
		rem = 90 - curr

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

			inc = rem / n
			bar.progress((curr + inc)/100)			
			
		bar.progress(90)

		pred_summary = summarize_input(
			summs, 
			summarization_model, 
			summarization_tokenizer,
			max_length=max_length,
			min_length=min_length,
		)

		bar.progress(100)

		end_time = perf_counter()
		time_taken = end_time - start_time

	with tmp_out.container():
		st.write(f"Took {time_taken} seconds")
		st.write(f"Summary: {pred_summary}")
	
	START = False
	COMPLETED = True

else:
	pass