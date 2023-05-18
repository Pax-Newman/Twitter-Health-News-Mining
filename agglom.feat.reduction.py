from models.embedders import FastText
from sklearn import cluster
from nltk.corpus import stopwords
from re import sub
from tqdm import tqdm

import nltk
import pandas as pd
import numpy as np
import argparse
import itertools
import pickle

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances



# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
import itertools
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as tokenize
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
stemIgnStop = SnowballStemmer('english', ignore_stopwords=True)
lemmer = WordNetLemmatizer()




parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/dataset.csv', help='path of the csv file to use')
parser.add_argument('--save_path', type=str, default='data/dataframe', help='path to place the resulting dataframe')
# parser.add_argument('--n_clusters', type=int, default=6, help='how many clusters to create with kmeans')
# parser.add_argument('--device', type=str, default='cpu', help='which device to use for embedding (applies only to bert)')

args = parser.parse_args()


# Load dataset
#df = pd.read_csv(args.data_path, delimiter='|', quoting=3)
df = pd.read_csv("data/dataset.csv")
# df = pd.read_pickle(args.data_path)


print(df.iloc[:5])

# Clean tweet content

# nltk.download('stopwords')
stopword_set = set([sub(r'[^\w\s]','',stop) for stop in stopwords.words('english')])


# used for fastText and TF-IDF
def clean(tweet: str) -> str:
    # Remove Unicode escape sequences (are these lines redundant?)
    cleaned = sub(r'â\S+>', '', tweet)
    cleaned = sub(r'â', '', cleaned)
    cleaned = sub(r'[\x00-\x1f\x7f-\xffÂ¼½ïð³]', '', cleaned) #some extra characters that slipped through the first two lines
    # Remove links
    cleaned = sub(r'http\S+', '', cleaned)
    # Remove usernames
    cleaned = sub(r'@\S+', '', cleaned)
    cleaned = cleaned.lower()
    # Remove punctuation
    cleaned = sub(r'[^\w\s]', '', cleaned)
    # Remove numbers
    cleaned = sub(r'[\d]','', cleaned)
    # Remove stopwords
    cleaned = ' '.join([word for word in cleaned.split(' ') if word not in stopword_set])
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

wordList = np.unique(list(itertools.chain(*[df["cleaned"].iloc[j,].split(sep = " ") for j in range(df.shape[0]-1)])), return_counts = True)
words = wordList[0][6:]


def syn_sim_metric(x, y):
    maxSim = 0
    for s1,s2 in itertools.product(wordnet.synsets(words[int(x[0])]), wordnet.synsets(words[int(y[0])])):
        if s1.pos() == s2.pos():
            score = wordnet.wup_similarity(s1,s2)
            if score>maxSim:
                maxSim = score
    return maxSim

words_index = np.arange(len(words)).reshape(-1, 1)

dist_mat = pairwise_distances(words_index, words_index, metric = syn_sim_metric)

np.save("word_dist_mat.npy", dist_mat, allow_pickle = True)
np.save("words.npy", words, allow_pickle = True)








