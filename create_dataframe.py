from embedders import FastText
from sklearn import cluster
from nltk.corpus import stopwords
from re import sub

import nltk
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/dataset.csv')

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

df = df[df['cleaned'] != '']

df.drop(df[df['cleaned'] == ''].index, inplace=True)

print(df[df['cleaned'] == ''])

print(len(df))

# Embed tweet content
embedder = FastText()

print('embedding content')
df['embedding'] = df['cleaned'].apply(embedder)

print(df.iloc[:5])

# Cluster the tweet embeddings
print('clustering embeddings')
df['label'] = cluster.KMeans(n_clusters=6).fit_predict(list(df['embedding']))

print(df.iloc[0])

# Save the dataframe for later use
print('pickling dataframe')
df.to_pickle('data/dataframe')


