#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as tokenize
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
stemIgnStop = SnowballStemmer('english', ignore_stopwords=True)
lemmer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


#load data
datafile = "data/dataset.csv"
twitter = pd.read_csv(datafile)


#removes usernames, links, and converts to lowercase
def remUserLinkCase(tweets):
    return tweets.str.replace(r'http\S+','',regex=True).str.replace(r'@\S+','',regex=True).str.lower()

#removes punctuation
def remPunct(tweets):
    return tweets.str.replace(r'[^\w\s]','', regex=True)

#removes stop words
def remStopWords(tweets):
    for i in range(0,len(tweets)):
        tweets[i] = ' '.join([w for w in tokenize(tweets[i]) if not w in stop_words])
    return tweets

#lemmatizes words; optional to remove or leave stop words
def lem(tweets, remStop = True):
    if remStop == True:
        for i in range(0,len(tweets)):
            tweets[i] = ' '.join([lemmer.lemmatize(w) for w in tokenize(tweets[i]) if not w in stop_words])
        return tweets
    else:
        for i in range(0,len(tweets)):
            tweets[i] = ' '.join([lemmer.lemmatize(w) for w in tokenize(tweets[i])])
        return tweets
    
#stems words; optional to remove or leave stop words
def stem(tweets, remStop = True):
    if remStop == True:
        for i in range(0,len(tweets)):
            tweets[i] = ' '.join([stemmer.stem(w) for w in tokenize(tweets[i]) if not w in stop_words])
        return tweets
    else:
        for i in range(0,len(tweets)):
            tweets[i] = ' '.join([stemmer.stem(w) for w in tokenize(tweets[i])])
        return tweets


#Examples/tests:
'''
stem(remPunct(remUserLinkCase(twitter["content"])), remStop = True)

stem(remPunct(remUserLinkCase(twitter["content"])), remStop = False)

lem(remPunct(remUserLinkCase(twitter["content"])), remStop = True)

lem(remPunct(remUserLinkCase(twitter["content"])), remStop = False)
'''

