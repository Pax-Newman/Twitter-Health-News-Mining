from embedders import FastText
from sklearn import cluster
from sklearn import decomposition
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from csv import reader
import pandas as pd

with open('data/dataset.csv') as f:
    datareader = reader(f)
    data = [
            row[3] for row in datareader
            ]

print('initializing model')
ft = FastText()

print(f'{len(data) = }')

print('embedding tweets')
embeddings = np.array([
    ft(tweet) for tweet in data[:]
        ])

print('reducing')
# reduced = decomposition.PCA(2).fit_transform(embeddings)
reduced = TSNE(verbose=1, n_jobs=-1).fit_transform(embeddings)

print('clustering')
#labels = cluster.KMeans(n_clusters=10).fit_predict(reduced)
labels = cluster.KMeans(n_clusters=6).fit_predict(embeddings)

df = pd.dataframe

# plt.scatter(reduced[:,0], reduced[:,1], c=labels)
#
# plt.savefig('kmeans')


# Create elbow graph 
# wcss = []
# for i in range(1, 20):
#     inertia = cluster.KMeans(n_clusters=i).fit(reduced).inertia_
#     wcss.append(inertia)
#
# plt.plot(range(1,20),wcss)
# plt.title('The Elbow Method Graph')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.savefig('elbow')




