from embedders import FastText
from sklearn import cluster

import numpy as np
import pandas as pd

df = pd.read_csv('data/dataset.csv')

print(df.iloc[:5])

embedder = FastText()


print('embedding content')

df['embedding'] = df['content'].apply(embedder)

print(df.iloc[:5])

print('clustering embeddings')
df['label'] = cluster.KMeans(n_clusters=6).fit_predict(list(df['embedding']))

print(df.iloc[0])

print('pickling dataframe')
df.to_pickle('data/dataframe')


