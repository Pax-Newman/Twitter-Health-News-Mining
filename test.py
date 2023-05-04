from models.embedders import FastText
from sklearn import cluster
from sklearn import decomposition
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from csv import reader
import pandas as pd
from re import sub
import torch
from models.reduction_net import ReductionNet
import nltk
from nltk.corpus import stopwords

# df = pd.read_csv('data/dataset.csv')
# # print([i for i in df[df['content'].str.contains('ò')]['content']])
# # quit()
#
# with open('data/dataset.csv') as f:
#     datareader = reader(f)
#     data = [
#             row[3] for row in datareader
#             ]
#
#
#

nltk.download('stopwords')
stopword_set = set(stopwords.words('english'))

def clean(tweet: str) -> str:
    # cleaned = tweet
    # Remove Unicode escape sequences
    cleaned = sub(r'â\S+>', '', tweet)
    # Remove links
    # cleaned = sub(r'http\S+', '', cleaned)
    # Remove usernames
    # cleaned = sub(r'@\S+', '', cleaned)
    # Remove hashtags
    cleaned = sub(r'#\S+', '', cleaned)
    # Remove punctuation
    cleaned = sub(r'[^\w\s]', '', cleaned)
    # Remove stopwords
    cleaned = ' '.join([word for word in cleaned.split(' ') if word not in stopword_set])
    return cleaned


df = pd.read_pickle('data/bertframe')

df['cleaned'] = df['content'].apply(clean)

print('initializing model')
ft = FastText()


df['fasttext'] = df['cleaned'].apply(ft)

df.to_pickle('data/bigframe')

#
# print(f'{embeddings.shape = }') # (63029, 300)

# df = pd.read_pickle('data/dataframe')
#
# df = df.rename(columns={'embedding':'bert', 'label':'bert label'})
#
# df.to_pickle('data/bertframe')

print(df.columns)
quit()

# print('reducing')
reduced = decomposition.PCA(2).fit_transform(list(df['embedding']))
# reduced = TSNE(verbose=1, n_jobs=-1).fit_transform(embeddings)
#
# print('clustering')
# #labels = cluster.KMeans(n_clusters=10).fit_predict(reduced)
# labels = cluster.KMeans(n_clusters=6).fit_predict(embeddings)
#
# df = pd.read_csv('data/dataset.csv')
# df['label'] = labels
#
#
# plt.scatter(reduced[:,0], reduced[:,1], c=labels)
#
# plt.savefig('kmeans')

# model = ReductionNet()

# Create elbow graph 
wcss = []
for i in range(1, 20):
    inertia = cluster.KMeans(n_clusters=i).fit(list(df['embedding'])).inertia_
    wcss.append(inertia)

plt.plot(range(1,20),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow')




