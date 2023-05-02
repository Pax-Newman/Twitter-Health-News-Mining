from embedders import FastText
from sklearn import cluster
from nltk.corpus import stopwords
from re import sub

import nltk
import pandas as pd

# Load dataset
df = pd.read_csv('data/dataset.csv')

print(df.iloc[:5])

# Clean tweet content

nltk.download('stopwords')
stopword_set = set(stopwords.words('english'))

def clean(tweet: str) -> str:
    # Remove links
    cleaned = sub(r'http\S+', '', tweet)
    # Remove usernames
    cleaned = sub(r'@\S+', '', tweet)
    # Remove hashtags
    cleaned = sub(r'#\S+', '', tweet)
    # Remove punctuation
    cleaned = sub(r'[^\w\s]'. '', tweet)
    # Remove stopwords
    cleaned = ' '.join([word for word in cleaned.split(' ') if word not in stopword_set])
    return cleaned

df['content'] = df['content'].apply(clean)

# Embed tweet content
embedder = FastText()

print('embedding content')
df['embedding'] = df['content'].apply(embedder)

print(df.iloc[:5])

# Cluster the tweet embeddings
print('clustering embeddings')
df['label'] = cluster.KMeans(n_clusters=6).fit_predict(list(df['embedding']))

print(df.iloc[0])


# Save the dataframe for later use
print('pickling dataframe')
df.to_pickle('data/dataframe')


