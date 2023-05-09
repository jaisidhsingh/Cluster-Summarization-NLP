from data import LargeArticleDataset

dataset = LargeArticleDataset()
sample = dataset.get_sample(0)
print(sample.keys())