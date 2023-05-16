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
import itertools
import pickle



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

import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process



'''
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/dataset.csv', help='path of the csv file to use')
parser.add_argument('--save_path', type=str, default='data/dataframe', help='path to place the resulting dataframe')
parser.add_argument('--n_clusters', type=int, default=6, help='how many clusters to create with kmeans')
parser.add_argument('--device', type=str, default='cpu', help='which device to use for embedding (applies only to bert)')

args = parser.parse_args()
'''

# Load dataset
#df = pd.read_csv(args.data_path, delimiter='|', quoting=3)
df = pd.read_csv("data/dataset.csv")

print(df.iloc[:5])

# Clean tweet content

# nltk.download('stopwords')
stopword_set = set([sub(r'[^\w\s]','',stop) for stop in stopwords.words('english')])

######################################################
# cleaning functions (regexes)
######################################################

def cleanTags(tweet: str) -> str:
    # Remove hashtags
    cleaned = sub(r'#\S+', '', tweet)
    return cleaned

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

# used for fastText and TF-IDF
def keepStop(tweet: str) -> str:
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
    # cleaned = ' '.join([word for word in cleaned.split(' ') if word not in stopword_set])
    return cleaned

# used for BERT
def min_clean(tweet: str) -> str:
    cleaned = sub(r'http\S+', '', tweet)
    cleaned = sub(r'â\S+>', '', cleaned)
    cleaned = sub(r'â', '', cleaned)
    cleaned = sub(r'[\x00-\x1f\x7f-\xffÂ¼½ïð³]', '', cleaned) #some extra characters that slipped through the first two lines
    cleaned = cleaned.lower()
    return cleaned


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


''' # not really used anymore
#####################################################################
# fuzzy feature reduction functions (for TF-IDF feature reduction
#####################################################################

#input list of noun phrases, list of unique elements (in descending order of avg. fuzz ratio), and dictionary
#replaces similar strings (reduces unique elements)
#adds replaced strings into the dictionary with their replacements as an (old string, new string) tuple
def fuzzify(keys, featDict,thresh):
    done=[]
    for i in range(len(keys)):
        for x in range(i+1,len(keys)):
            if not(keys[x] in done):
                ratio=fuzz.ratio(keys[i],keys[x])
            if ratio>thresh:
                print(keys[x]," : ", keys[i])
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
                    keyDict[features[i]]=0fu
                if not(ratio==100):
                    keyDict[features[i]]=keyDict[features[i]]+ratio
                    count=count+1
            finList.append(features[i])
            keyDict[features[i]]=keyDict[features[i]]/count
        if(i%1000==0):
            print("i: ", i)


#input a dictionary with (string, float) tuples
#returns a list of strings in descending order of their corresponding float
def sortKeys(keyDict):
  allVals=[keyDict.get(k) for k in keyDict]
  sortedVals=sorted(allVals,reverse=True)
  orderedFeat=[list(keyDict.keys())[list(keyDict.values()).index(x)] for x in sortedVals]
  return orderedFeat

'''





# Init tqdm for pandas df.apply
tqdm.pandas()

# Remove hashtags
#df['noTags'] = df['content'].apply(cleanTags)

# Clean the content
df['cleaned'] = df['content'].apply(clean)
df['keepStop'] = df['content'].apply(keepStop)
#df['cleanedNoTags'] = df['noTags'].apply(clean)

print(f'current size: {len(df)}')

# Drop rows with empty strings
df = df.drop(df[df['cleaned'] == ''].index)
# Drop rows with only whitespace
df = df.drop(df[df['cleaned'].str.isspace()].index)

print(f'size after cleaning: {len(df)}')

wordsWithStop = np.unique(list(itertools.chain(*[df["keepStop"].iloc[j,].split(sep = " ") for j in range(df.shape[0]-1)])))
wordsNoStop = np.unique(list(itertools.chain(*[df["cleaned"].iloc[j,].split(sep = " ") for j in range(df.shape[0]-1)])))
print('number of words with stop words: ', len(wordsWithStop))
print('number of words without stop words: ', len(wordsNoStop))




# Feature reduction for words in tweets
wordList = np.unique(list(itertools.chain(*[df["cleaned"].iloc[j,].split(sep = " ") for j in range(df.shape[0]-1)])), return_counts = True)
words = wordList[0][6:]
counts = wordList[1][6:]
wordCounts = np.array([words, counts]).T
wordCountsSorted = wordCounts[wordCounts[:, 1].astype(int).argsort()]
wordCountDict = dict(zip(wordCounts[:,0],wordCounts[:,1].astype(int)))

wordsMoreThan2 = wordCounts[wordCounts[:,1].astype(int)>2]
print("number of words with frequency >2: ", wordsMoreThan2.shape[0])

lemmed = np.unique(lem(words))
stemmed = np.unique(stem(words))
lemmedMoreThan2 = np.unique(lem(wordsMoreThan2[:,0]))
stemmedMoreThan2 = np.unique(stem(wordsMoreThan2[:,0]))
print('number of unique words after lemmatizing: ', len(lemmed))
print('number of unique words after stemming: ', len(stemmed))
print('number of unique words with freq >2 after lemmatizing: ', len(lemmedMoreThan2))
print('number of unique words with freq >2 after stemming: ', len(stemmedMoreThan2))

swapDict = {}
for i in range(len(words)):
    w = wordCountsSorted[len(words)-(i+1),0]
    max_score = .85
    swap = ''
    for k in wordCountsSorted[len(words)-i:,0]:
        for s1, s2 in itertools.product(wordnet.synsets(w), wordnet.synsets(k)):
            score = wordnet.wup_similarity(s1,s2)
            if score > max_score:
                max_score = score
                swap = k
    if swap != '':
        print('w: ', w, ' swap: ', swap)
        while swap in swapDict:
            swap = swapDict[swap]
        swapDict[w] = swap
        print('final swap: ', swap)
        print()

print('number of features reduced using synonyms: ', len(swapDict.keys()))



''' 
#this was all a little bit silly
stemlemmed = np.unique(stem(lemmed))
lemstemmed = np.unique(lem(stemmed))
print('number of unique words after stemlemmatizing: ', len(stemlemmed))
print('number of unique words after lemstemming: ', len(lemstemmed))

# this took a while and wasn't that useful because string similarity is limited and 
# unhelpful for generalizing to semantic relationships when working with words 
# (rather than longer strings with more letters to compare)

keyDictLem = {}
wordDictLem = {}
print('generating fuzzy dictionary')
simCount(lemmed, keyDictLem)
keySortLem=sortKeys(keyDictLem)
fuzzify(keySortLem,wordDictLem,.85)
print('number of words in fuzzy dictionary: ', len(wordDictLem))

print('pickling dictionary')
with open('wordDictLem.85.pickle', 'wb') as file:
    pickle.dump(wordDictLem, file)



# and this all was just too complicated and didn't end up working right
wordSyns = set(ss for w in words for ss in wordnet.synsets(w))

simScores = np.empty((0, 3))

# ran this the first time, took 30+ hours because it appends synsets objects into the np array
 for w1, w2 in itertools.combinations(wordSyns, 2):
     score = wordnet.wup_similarity(w1,w2)
     if score>0.6:
         simScores = np.append(simScores, [[score, w1, w2]], axis=0)

# better version of the loop (produces something equivalent to what i eventually saved and loaded from above)


#tried more/other stuff down here:
for w1, w2 in itertools.combinations(wordSyns,2):
    score = wordnet.wup_similarity(w1,w2)
    if score>0.8:
        simScores = np.append(simScores, [[score, w1.lemma_names()[0], w2.lemma_names()[0]]], axis = 0)

# sorted similarity scores
simSorted = simScores[simScores[:, 0].argsort()]

# just the words that appear as identifiers of different synonym maps
simWords = np.unique(np.append(simScores[:,1], simScores[:,2], axis = 0))

# for each word, sum all of its high similarity scores
# score is higher for words that are similar to a lot of other words, and for words that have higher similarities 
simSummed = np.empty((0,2))
for w in simWords:
    sum1 = sum(simScores[simScores[:,1]== w, 0].astype(float))
    sum2 = sum(simScores[simScores[:,2]== w, 0].astype(float))
    simSummed = np.append(simSummed, [[sum1+sum2, w]], axis = 0)

# kinda lost steam at this point couldn't figure out what/how to do
swapDict = {}
done = []
for score, s1, s2 in simSorted:
    if s1 not in done and s2 not in done:
        if simSummed[simSummed[:,1]==s1,0]>=simSummed[simSummed[:,1]==s2,0]:
            swapDict[s2]=s1
            done = done+[s2]
        else:
            swapDict[s1]=s2
            done = done+[s1]
''' 

            
           


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




























