from models.embedders import FastText
from sklearn import cluster
from nltk.corpus import stopwords
from re import sub
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import nltk
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/dataset.csv', help='path of the csv file to use')
parser.add_argument('--save_path', type=str, default='data/dataframe', help='path to place the resulting dataframe')
parser.add_argument('--n_clusters', type=int, default=6, help='how many clusters to create with kmeans')
parser.add_argument('--embedder', type=str, default='fasttext', choices=['fasttext', 'bert'] , help='which embedder to use')
parser.add_argument('--device', type=str, default='cpu', help='which device to use for embedding (applies only to bert)')

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

# Init tqdm for pandas df.apply
tqdm.pandas()

if args.embedder == 'fasttext':    
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
    df['embedding'] = df['cleaned'].progress_apply(embedder)
elif args.embedder == 'bert':
    embedder = SentenceTransformer('sentence-transformers/all-roberta-large-v1', device=args.device).encode
    
    df['embedding'] = df['content'].progress_apply(embedder)
    # df['embedding'] = embedder.encode(tqdm(df['content']), batch_size=32)

print(df.iloc[:5])

# Cluster the tweet embeddings
print('clustering embeddings')
df['label'] = cluster.KMeans(n_clusters=args.n_clusters).fit_predict(list(df['embedding']))

print(df.iloc[0])

# Save the dataframe for later use
print('pickling dataframe')
df.to_pickle(args.save_path)


