from models.embedders import FastText
from sklearn import cluster
from nltk.corpus import stopwords
from re import sub

import nltk
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/dataset.csv')
parser.add_argument('--save_path', type=str, default='data/dataframe')
parser.add_argument('--n_clusters', type=int, default=6)

args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.data_path)

print(df.iloc[:5])

# Clean tweet content

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

df['cleaned'] = df['content'].apply(clean)

print(df.iloc[:5]['cleaned'])

print('\nEmpty cleaned rows:')
print(df[df['cleaned'] == ''])

print(len(df))


df = df.drop(df[df['cleaned'] == ''].index)

df = df[df['cleaned'] != '']

print(df[df['cleaned'] == ''])

print(len(df))

# Embed tweet content
embedder = FastText()

print('embedding content')
df['embedding'] = df['cleaned'].apply(embedder)

print(df.iloc[:5])

# Cluster the tweet embeddings
print('clustering embeddings')
df['label'] = cluster.KMeans(n_clusters=args.n_clusters).fit_predict(list(df['embedding']))

print(df.iloc[0])

# Save the dataframe for later use
print('pickling dataframe')
df.to_pickle(args.save_path)


