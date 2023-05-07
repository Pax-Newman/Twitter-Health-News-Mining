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
parser.add_argument('--device', type=str, default='cpu', help='which device to use for embedding (applies only to bert)')

args = parser.parse_args()

# Load dataset
#df = pd.read_csv(args.data_path, delimiter='|', quoting=3)
df = pd.read_pickle(args.data_path)

print(df.iloc[:5])

# Clean tweet content

nltk.download('stopwords')
stopword_set = set(stopwords.words('english'))

def clean(tweet: str) -> str:
    # Remove Unicode escape sequences
    cleaned = sub(r'â\S+>', '', tweet)
    cleaned = sub(r'â', '', cleaned)
    # Remove links
    cleaned = sub(r'http\S+', '', cleaned)
    # Remove usernames
    cleaned = sub(r'@\S+', '', cleaned)
    # Remove hashtags
    # cleaned = sub(r'#\S+', '', cleaned)
    # Remove punctuation
    cleaned = sub(r'[^\w\s]', '', cleaned)
    # Remove stopwords
    cleaned = ' '.join([word for word in cleaned.split(' ') if word not in stopword_set])
    return cleaned

def min_clean(tweet: str) -> str:
    cleaned = sub(r'http\S+', '', tweet)
    cleaned = sub(r'â\S+>', '', cleaned)
    cleaned = sub(r'â', '', cleaned)
    return cleaned

# Init tqdm for pandas df.apply
tqdm.pandas()

# Clean the content
df['cleaned'] = df['content'].apply(clean)

print(f'current size: {len(df)}')

# Drop rows with empty strings
df = df.drop(df[df['cleaned'] == ''].index)
# Drop rows with only whitespace
df = df.drop(df[df['cleaned'].str.isspace()].index)

print(f'size after cleaning: {len(df)}')

# Create FastText embeddings
ft = FastText()
print('generating FastText embeddings')

df['fasttext'] = df['cleaned'].progress_apply(ft)

# Create BERT embeddings
bert = SentenceTransformer('sentence-transformers/all-roberta-large-v1', device=args.device).encode
print('generating BERT embeddings')

df['bert'] = df['content'].progress_apply(lambda tweet : bert(min_clean(tweet)))

# Cluster the tweet embeddings??

# Save the dataframe for later use
print('pickling dataframe')
df.to_pickle(args.save_path)


