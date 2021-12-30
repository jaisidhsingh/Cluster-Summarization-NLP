# Multi-Article Summarization Using Sentence Clustering

### Dependancies:

> pytorch

> transformers (huggingface)

> sentence-transformers (SentenceBERT)

> numpy

> sklearn

> matplotlib

> nltk (Natural Language Toolkit)

### Usage:

```
python3 summarize.py
```

The above command loads in an example article set (check 'examples' folder)
To enter your own articles add the argument

```
python3 summarize.py --custom True
```

This starts a simple terminal input loop to enter the articles. Type 'exit' to submit and trigger summarization.

### Approach:
For multiple articles, the sentences are first turned to embeddings and resulting clusters in the embedding space are extracted. Finally each cluster is summarized.

![tSNE](https://user-images.githubusercontent.com/75247817/147751106-2acecd28-eadc-43f7-b440-bb664f0454bf.png)

An example plot projecting clusters of sentence embeddings in 384 dimensions in 2D via tSNE (outputs for each run can be visualized using either tSNE or PCA, using 'visualize_outputs' method of the 'ClusterSentenceEmbeddings' class in 'src/utils.py'


Web-app for better hyperparameter access is very likely in the future.
