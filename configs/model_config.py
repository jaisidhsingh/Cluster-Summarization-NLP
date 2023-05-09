from types import SimpleNamespace


cfg = SimpleNamespace(**{})

# sentence embedding model configs
cfg.sent_model_name = "sentence-transformers/all-MiniLM-L6-v2"
cfg.sent_model_seq_limit = 256

# summarization model configs

