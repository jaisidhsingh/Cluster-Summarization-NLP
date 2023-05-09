from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_summarizer(model_code):
    name_dict = {
        "bart": "facebook/bart-large-cnn",
        "distill-bart": "sshleifer/distilbart-cnn-12-6",
        "roberta": "google/roberta2roberta_L-24_cnn_daily_mail",
        "pegasus": "google/pegasus-cnn_dailymail"
    }

    model_name = name_dict[model_code.lower()]
    model, tokenizer = None, None
    
    if "bart" in model_name:
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
    
    if "pegasus" in model_name:
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

    if "roberta" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer

def summarize_input(
        input_article, 
        model, 
        tokenizer,
        max_length=150,
        min_length=50,
        num_beams=3,
        length_penalty=0.5,
        no_repeat_ngram_size=3
    ):
    text_input_ids = tokenizer.batch_encode_plus(
        [input_article], 
        return_tensors='pt', 
        max_length=tokenizer.model_max_length
    )['input_ids'].to("cpu")
    
    summary_ids = model.generate(
        text_input_ids, 
        num_beams=int(num_beams), 
        length_penalty=float(length_penalty), 
        max_length=int(max_length), 
        min_length=int(min_length), 
        no_repeat_ngram_size=int(no_repeat_ngram_size)
    )
    summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary_txt.replace("<n>", "")