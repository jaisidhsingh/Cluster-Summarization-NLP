from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')
SUMMARIZATION_TOKENIZER = BartTokenizer.from_pretrained(
    'facebook/bart-large-cnn')
SUMMARIZATION_MODEL = BartForConditionalGeneration.from_pretrained(
    'facebook/bart-large-cnn')
