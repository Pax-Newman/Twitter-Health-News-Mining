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

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as tokenize
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
stemIgnStop = SnowballStemmer('english', ignore_stopwords=True)
lemmer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


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

######################################################
# cleaning functions (regexes)
######################################################

def hashClean(tweet: str) -> str:
    # Remove hashtags
        cleaned = sub(r'#\S+', '', cleaned)

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
    # Remove punctuation
    cleaned = sub(r'[^\w\s]', '', cleaned)
    # Remove numbers
    cleaned = sub(r'[\d]','', cleaned)
    # Remove stopwords
    cleaned = ' '.join([word for word in cleaned.split(' ') if word not in stopword_set])
    return cleaned

# used for BERT
def min_clean(tweet: str) -> str:
    cleaned = sub(r'http\S+', '', tweet)
    cleaned = sub(r'â\S+>', '', cleaned)
    cleaned = sub(r'â', '', cleaned)
    return cleaned


####################################################
# functions for lemmatizing and stemming
####################################################

#lemmatizes words; optional to remove or leave stop words
def lem(tweets):
    for i in range(0,len(tweets)):
       tweets[i] = ' '.join([lemmer.lemmatize(w) for w in tokenize(tweets[i])])
    return tweets
    
#stems words; optional to remove or leave stop words
def stem(tweets):
    for i in range(0,len(tweets)):
       tweets[i] = ' '.join([stemmer.stem(w) for w in tokenize(tweets[i])])
    return tweets



#####################################################################
# fuzzy feature reduction functions (for TF-IDF feature reduction
#####################################################################

#input list of noun phrases, list of unique elements (in descending order of avg. fuzz ratio), and dictionary
#replaces similar strings (reduces unique elements)
#adds replaced strings into the dictionary with their replacements as an (old string, new string) tuple
def fuzzify(keys, featDict):
  done=[]
  for i in range(len(keys)):
    for x in range(i+1,len(keys)):
      if not(keys[x] in done):
        ratio=fuzz.ratio(keys[i],keys[x])
        if ratio>90:
          featDict[keys[x]]=keys[i]
          keys[x]=keys[i]
    done.append(keys[i])

#takes list of noun phrases, adds dictionary entries to keyDict with (noun phrase, avg. fuzz.ratio score)
#fuzz.ratio average is calculated using all non-identical elements
#returns a list of unique elements from the inputed list
#an equal number of tuples are added to keyDict as are in the returned list (every unique value is given a dictionary entry)
def simCount(features,keyDict):
  finList=[]
  for i in range(len(features)):
    if not(features[i] in finList): 
      count=0
      for x in range(len(features)):
        ratio=fuzz.ratio(features[i],features[x])
        if not(features[i] in keyDict):
          keyDict[features[i]]=0
        if not(ratio==100):
          keyDict[features[i]]=keyDict[features[i]]+ratio
          count=count+1
      finList.append(features[i])
      keyDict[features[i]]=keyDict[features[i]]/count

#input a dictionary with (string, float) tuples
#returns a list of strings in descending order of their corresponding float
def sortKeys(keyDict):
  allVals=[keyDict.get(k) for k in keyDict]
  sortedVals=sorted(allVals,reverse=True)
  orderedFeat=[list(keyDict.keys())[list(keyDict.values()).index(x)] for x in sortedVals]
  return orderedFeat







# Init tqdm for pandas df.apply
tqdm.pandas()

# Remove hashtags
df['noTags'] = df['content'].apply(cleanTags)

# Clean the content
df['cleaned'] = df['content'].apply(clean)
df['cleanedNoTags'] = df['noTags'].apply(clean)

print(f'current size: {len(df)}')

# Drop rows with empty strings
df = df.drop(df[df['cleaned'] == ''].index)
# Drop rows with only whitespace
df = df.drop(df[df['cleaned'].str.isspace()].index)

print(f'size after cleaning: {len(df)}')




# Feature reduction for words in tweets
words = np.unique(list(itertools.chain(*[df["cleaned"].iloc[j,].split(sep = " ") for j in range(data.shape[0]-1)])))
lemmed = np.unique(lem(words))
stemmed = np.unique(stem(words))
print('number of unique words: ', len(words))
print('number of unique words after lemmatizing: ', len(lem))
print('number of unique words after stemming: ', len(stem))

keyDict = {}
wordDict = {}
print('generating fuzzy dictionary')
simCount(words, keyDict)
keySort=sortKeys(keyDict)
fuzzify(keySort,wordDict)
print('number of words in fuzzy dictionary: ', len(wordDict))




# Create FastText embeddings
ft = FastText()
print('generating FastText embeddings')

df['fast'] = df['cleaned'].progress_apply(ft)
df['fastNoTags'] = df['cleanedNoTags'].progress_apply(ft)

# Create BERT embeddings
bert = SentenceTransformer('sentence-transformers/all-roberta-large-v1', device=args.device).encode
print('generating BERT embeddings')

df['bert'] = df['content'].progress_apply(lambda tweet : bert(min_clean(tweet))))
df['bertNoTags'] = df['noTags'].progress_apply(lambda tweet : bert(min_clean(tweet))))

# Cluster the tweet embeddings??

# Save the dataframe for later use
print('pickling dataframe')
df.to_pickle(args.save_path)


