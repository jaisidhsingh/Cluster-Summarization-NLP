import os
import numpy as np


def example_set(set=1):
    assert set == 1, f'Set{set} was not found among the provided example sets'

    articles = []
    path = f"examples/set{set}/"
    for file in os.listdir(path):
        f = open(path+file, "r", encoding='utf8')
        data = f.read()
        data.replace("\n", "")
        articles.append(data)
        f.close()

    return np.array(articles)
