from models.embedders import FastText
from sklearn import cluster
from nltk.corpus import stopwords
from re import sub
# from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import pandas as pd
import numpy as np
import argparse
import itertools
import pickle


''' # not needed unless you're regenerating wordnet stuff or lemmatizing/stemming
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as tokenize
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
stemIgnStop = SnowballStemmer('english', ignore_stopwords=True)
lemmer = WordNetLemmatizer()
'''

''' # re-enable to use command line args
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/dataset.csv', help='path of the csv file to use')
parser.add_argument('--save_path', type=str, default='data/dataframe', help='path to place the resulting dataframe')
parser.add_argument('--n_clusters', type=int, default=6, help='how many clusters to create with kmeans')
parser.add_argument('--device', type=str, default='cpu', help='which device to use for embedding (applies only to bert)')

args = parser.parse_args()
'''

# Load dataset
#df = pd.read_csv(args.data_path)
df = pd.read_csv("data/dataset.csv")

print(df.iloc[:5])

# nltk.download('stopwords')
stopword_set = set([sub(r'[^\w\s]','',stop) for stop in stopwords.words('english')])
# reducDict of word replacement for TF-IDF feature reduction
with open('reducDict.pickle', 'rb') as doc:
    reducDict = pickle.load(doc)

######################################################
# cleaning functions (regexes)
######################################################

# used for fastText and TF-IDF
def clean(tweet: str) -> str:
    cleaned = tweet.lower()
    # Remove Unicode escape sequences (are these lines redundant?)
    cleaned = sub(r'â\S+>', '', cleaned)
    cleaned = sub(r'â', '', cleaned)
    cleaned = sub(r'[\x00-\x1f\x7f-\xffÂ¼½ïð³]', '', cleaned) #some extra characters that slipped through the first two lines
    # Remove links
    cleaned = sub(r'http\S+', '', cleaned)
    # Remove usernames
    cleaned = sub(r'@\S+', '', cleaned)
    # Remove punctuation
    cleaned = sub(r'[^\w\s]', '', cleaned)
    # Remove numbers
    cleaned = sub(r'[\d]','', cleaned)
    # Remove stopwords
    cleaned = ' '.join([word for word in cleaned.split(' ') if word not in stopword_set])
    return cleaned.strip()


# same as above but leaves in stop words
def keepStop(tweet: str) -> str:
    cleaned = tweet.lower()
    # Remove Unicode escape sequences (are these lines redundant?)
    cleaned = sub(r'â\S+>', '', cleaned)
    cleaned = sub(r'â', '', cleaned)
    cleaned = sub(r'[\x00-\x1f\x7f-\xffÂ¼½ïð³]', '', cleaned) #some extra characters that slipped through the first two lines
    # Remove links
    cleaned = sub(r'http\S+', '', cleaned)
    # Remove usernames
    cleaned = sub(r'@\S+', '', cleaned)
    # Remove punctuation
    cleaned = sub(r'[^\w\s]', '', cleaned)
    # Remove numbers
    cleaned = sub(r'[\d]','', cleaned)
    # Remove stopwords
    # cleaned = ' '.join([word for word in cleaned.split(' ') if word not in stopword_set])
    return cleaned.strip()


# used for TF-IDF
def featReduc(tweet: str) -> str:
    reduced = ' '.join([reducDict[word] if word in reducDict else word for word in tweet.split(' ')]) # replaces words with similar words from reducDict
    return reduced.strip()

def removeSingle(tweet: str, singleList) -> str:
    cleaned = ' '.join([word for word in tweet.split(' ') if word not in singleList]) # removes words that only occur once in the corpus
    return cleaned

# used for BERT
def min_clean(tweet: str) -> str:
    cleaned = tweet.lower()
    cleaned = sub(r'http\S+', '', cleaned)
    cleaned = sub(r'â\S+>', '', cleaned)
    cleaned = sub(r'â', '', cleaned)
    cleaned = sub(r'[\x00-\x1f\x7f-\xffÂ¼½ïð³]', '', cleaned) #some extra characters that slipped through the first two lines
    return cleaned.strip()


''' # only used for collecting information about feature reduction
####################################################
# functions for lemmatizing and stemming
####################################################

#lemmatizes words; optional to remove or leave stop words
def lem(words):
    for i in range(0,len(words)):
       words[i] = lemmer.lemmatize(words[i])
    return words
    

#stems words; optional to remove or leave stop words
def stem(words):
    for i in range(0,len(words)):
       words[i] = stemmer.stem(words[i])
    return words
'''

# Init tqdm for pandas df.apply
tqdm.pandas()

# Clean the content
df['cleaned'] = df['content'].apply(clean) # for fastText and TF-IDF
df['keepStop'] = df['content'].apply(keepStop)  # for comparison/analysis
df['featReduc'] = df['cleaned'].apply(featReduc)   # initial feature reduction for TF-IDF
df['minClean'] = df['content'].apply(min_clean)    # for bert

# Drop rows with empty strings
df = df.drop(df[df['cleaned'] == ''].index)
# Drop rows with only whitespace
df = df.drop(df[df['cleaned'].str.isspace()].index)

cleanedWords = np.unique(list(itertools.chain(*[df["cleaned"].iloc[j,].split(sep = " ") for j in range(df.shape[0]-1)])), return_counts = True)
keepStopWords = np.unique(list(itertools.chain(*[df["keepStop"].iloc[j,].split(sep = " ") for j in range(df.shape[0]-1)])), return_counts = True)
featReducWords = np.unique(list(itertools.chain(*[df["featReduc"].iloc[j,].split(sep = " ") for j in range(df.shape[0]-1)])), return_counts = True)

'''
cleanedWords[0][0:7] # for some reason the first 6 entries here are empty/underscores... we'll ignore those
# array(['', '___', '_____', '______', '_________', '_____________', 'aa'])
keepStopWords[0][0:7] # same thing happens with other two types
# array(['', '___', '_____', '______', '_________', '_____________', 'a'])
featReducWords[0][0:7]
# array(['', '___', '_____', '______', '_________', '_____________', 'aa'])
'''

# first column is the word; second column is the number of occurances
cleanedWordCounts = np.array([cleanedWords[0][6:], cleanedWords[1][6:]]).T
cleanedWordCounts.shape # (30146, 2)
keepStopWordCounts = np.array([keepStopWords[0][6:], keepStopWords[1][6:]]).T
keepStopWordCounts.shape # (30304, 2)
featReducWordCounts = np.array([featReducWords[0][6:], featReducWords[1][6:]]).T
featReducWordCounts.shape # (17463, 2)

''' # more stuff that's just analysis
# number/proportion of stop words
sum(keepStopWordCounts[:,1].astype(int)) # 681039
sum([row[1].astype(int) for row in keepStopWordCounts if row[0] in stopword_set]) # 211824
# number of stop words removed from data: 30303 - 30145 = 158
# proportion of words which were stop words:  211824/681039　= 0.311

# number/proportion of feature reduced words
sum(cleanedWordCounts[:,1].astype(int)) # 469215
sum([row[1].astype(int) for row in cleanedWordCounts if row[0] not in featReducWordCounts[:,0]]) # 192857
# number of unique words replaced using feature reduction: 30146 - 17463 = 12683
# proportion of words which were replaced using feature reduction: 192857/469215 = 0.411

# number/proportion of words with a single occurance
sum(cleanedWordCounts[cleanedWordCounts[:,1].astype(int)==1][:,1].astype(int)) # 13214
sum(featReducWordCounts[featReducWordCounts[:,1].astype(int)==1][:,1].astype(int)) # 8873
# number of single-occurance words replaced using feature reduction: 13214 - 8873 = 4341 (4341/13214 = 0.33)
# old proportion of words which occurred once (before feature reduction): 13214/469215 = 0.028
# new proportion of words which occur once (after feature reduction): 8873/469215 = 0.018
'''

featReducSingleWords = featReducWordCounts[featReducWordCounts[:,1].astype(int)==1][:,0]
''' # more stuff that's just analysis
cleanedSingleWords = cleanedWordCounts[cleanedWordCounts[:,1].astype(int)==1][:,0]
len(set([tweet for tweet in df["cleaned"] for word in cleanedSingleWords if word in tweet.split(' ')])) # 10914
len(set([tweet for tweet in df["featReduc"] for word in featReducSingleWords if word in tweet.split(' ')])) # 7576
# number of tweets containing single occurrance words before featReduc: 10914 (proportion: 10914/63021 = 0.173)
# number of tweets containing single occurrance words after featReduc: 7576 (proportion: 7576/63021 = 0.120)
# number of tweets containing >1 single occurrance word before featReduc: 13214 - 10914 = 2300
# number of tweets containing >1 single occurrance word after featReduc: 8873 - 7576 = 1297
'''

# More cleaning (once singleReducWords has been defined)
df['finalReduc'] = df['featReduc'].apply(removeSingle, singleList = featReducSingleWords)   # more for TF-IDF

finalReducWords = np.unique(list(itertools.chain(*[df["finalReduc"].iloc[j,].split(sep = " ") for j in range(df.shape[0]-1)])), return_counts = True)
# finalReducWords[0][0:7] # still have the weird first 6 entries
# array(['', '___', '_____', '______', '_________', '_____________', 'aa'])
finalReducWordCounts = np.array([finalReducWords[0][6:], finalReducWords[1][6:]]).T

''' # more that's just analysis
# comparison of different feature reduction methods:
len(set(lem(cleanedWordCounts[:,0]))) # 26414
len(set(stem(cleanedWordCounts[:,0]))) # 20940
len(finalReducWordCounts[:,0]) # 8590
# number of unique words (in cleaned data): 30145
# number of unique words with more than one occurrance: 30145 - 13214 = 16931 (proportion: 16931/30145 = 0.562)
# number of unique words after lemmatizing: 26414 (proportion: 26414/30145 = 0.876)
# number of unique words after stemming: 20940 (proportion: 20940/30145 = 0.695)
# number of unique words after cluster feature reduction: 17463 (proportion: 17463/30145 = 0.579)
# number of unique words after cluster feature reduction and single word removal: 8590 (proportion: 8590/30145 = 0.285)
'''

# save data frame: columns saved are id, datetime, publisher, content, cleaned (for FastText), minClean (for bert), and finalReduc (for TF-IDF)
# df[['id','datetime','publisher','content','cleaned','minClean','finalReduc']].to_pickle('allTheTextData.pickle')

''' # can't run these on my computer
# Create FastText embeddings
ft = FastText()
print('generating FastText embeddings')
df['fast'] = df['cleaned'].progress_apply(ft)

# Create BERT embeddings
bert = SentenceTransformer('sentence-transformers/all-roberta-large-v1', device=args.device).encode
print('generating BERT embeddings')
df['bert'] = df['minClean'].progress_apply(bert)
'''

# Create TF-IDF embeddings
vectorizer = TfidfVectorizer(token_pattern = r"\S+", stop_words = ['___', '_____', '______', '_________', '_____________'])
tfidf_matrix = vectorizer.fit_transform(list(df['finalReduc']))
# i think something is broken in these next lines
# tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
# df['tfidf'] = tfidf_matrix.toarray()

# save data frame: columns saved are id, datetime, publisher, content, cleaned (for FastText), minClean (for bert), finalReduc (for TF-IDF), fast (fast text embedding), bert (bert embedding), and tfidf (tfidf embedding)
# df[['id','datetime','publisher','content','cleaned','minClean','finalReduc', 'fast', 'bert', 'tfidf']].to_pickle('allData.pickle')

