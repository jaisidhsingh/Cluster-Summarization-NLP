import os
import sys
cwd = os.getcwd()
module2add = '\\'.join(cwd.split("\\")[:-1])
sys.path.append(module2add)

from configs.model_config import cfg as model_configs

from transformers import AutoTokenizer, AutoModel
import torch


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def make_embeddings(sentence_list, pool_fn):
    tokenizer = AutoTokenizer.from_pretrained(model_configs.sent_model_name)
    model = AutoModel.from_pretrained(model_configs.sent_model_name)

    encoded_input = tokenizer(
        sentence_list, 
        padding=True, 
        truncation=True, 
        max_length=model_configs.sent_model_seq_limit, 
        return_tensors='pt'
    )
    with torch.no_grad():
        embeddings = model(**encoded_input)
    
    attn_mask = encoded_input['attention_mask']
    sentence_embeddings = pool_fn(embeddings, attn_mask)
    return sentence_embeddings

def test_embedder():
    sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']
    
    embeddings = make_embeddings(sentences)
    print(embeddings.shape)
