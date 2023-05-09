from models.summarizers import load_summarizer

out = load_summarizer("pegasus")
out = load_summarizer("distill-bart")
out = load_summarizer("roberta")