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

from tqdm import tqdm
import csv

from sentence_transformers import SentenceTransformer

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



nltk.download('stopwords')
stopword_set = set(stopwords.words('english'))

def clean(tweet: str) -> str:
    # cleaned = tweet
    # Remove Unicode escape sequences
    cleaned = sub(r'â\S+>', '', tweet)
    # Remove links
    cleaned = sub(r'http\S+', '', cleaned)
    # Remove usernames
    cleaned = sub(r'@\S+', '', cleaned)
    # Remove hashtags
    cleaned = sub(r'#\S+', '', cleaned)
    # Remove punctuation
    cleaned = sub(r'[^\w\s]', '', cleaned)
    # Remove stopwords
    cleaned = ' '.join([word for word in cleaned.split(' ') if word not in stopword_set])
    return cleaned

def remove_links(tweet: str) -> str:
    # Remove links
    cleaned = sub(r'http\S+', '', tweet)
    return cleaned

with open('Health-Tweets/cbchealth.txt') as f:
    print('yay')
quit()

df= pd.read_csv('Health-Tweets/cbchealth.txt', sep='|', quoting=csv.QUOTE_ALL)

for row in df['content']:
    print(row)
quit()

df = pd.read_pickle('data/bigframe')

wcss = []
for i in range(1, 35):
    print(f'{i = }')
    inertia = cluster.KMeans(n_clusters=i, n_init='auto').fit(list(df['bert'])).inertia_
    wcss.append(inertia)

plt.plot(range(1,35),wcss)
plt.title('Elbow Method Graph (BERT)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('bertelbow')

quit()

df['cleaned'] = df['content'].apply(clean)

df = df.drop(df[df['cleaned'] == ''].index)
df = df.drop(df[df['cleaned'].str.isspace()].index)

tqdm.pandas()

print('running fasttext')
ft = FastText()

df['fasttext'] = df['cleaned'].progress_apply(ft)

bert = SentenceTransformer('sentence-transformers/all-roberta-large-v1', device='mps').encode

print('running bert')
df['bert'] = df['content'].progress_apply(lambda t : bert(remove_links(t)))

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




