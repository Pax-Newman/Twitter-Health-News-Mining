#!/usr/bin/env python
# coding: utf-8

# In[7]:


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

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVec = TfidfVectorizer()


# In[2]:


datafile = "data/dataset.csv"
twitter = pd.read_csv(datafile)


# In[43]:


#removes usernames, links, and converts to lowercase
def remUserLinkCase(tweets, remHash = False):
    if remHash==True:
        return tweets.str.replace(r'http\S+','',regex=True).str.replace(r'@\S+','',regex=True).str.replace(r'#\S+','',regex=True).str.lower()
    else:
        return tweets.str.replace(r'http\S+','',regex=True).str.replace(r'@\S+','',regex=True).str.lower()

#removes punctuation
def remPunct(tweets, remNum = False):
    if remNum==True:
        return tweets.str.replace(r'[\d]','', regex=True).str.replace(r'[^\w\s]','', regex=True)
    else:
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


# In[4]:


cleaned = lem(remPunct(remUserLinkCase(twitter["content"])), remStop = True)


# In[36]:


stemmed = stem(remPunct(remUserLinkCase(twitter["content"])), remStop = True)


# In[44]:


noTagLem = lem(remPunct(remUserLinkCase(twitter["content"], remHash = True), remNum = False), remStop = True)


# In[46]:


noTagVec = tfidfVec.fit_transform(noTagLem.tolist())


# In[51]:


tfidfVec.get_feature_names_out().shape


# In[17]:


tfidf = tfidfVec.fit_transform(cleaned.tolist())


# In[37]:


stemmedidf = tfidfVec.fit_transform(stemmed.tolist())


# In[41]:


tfidfVec.get_feature_names_out()[0:1000]


# In[31]:


cleaned.tolist()


# In[27]:


cleaned[1]


# In[ ]:


stem(remPunct(remUserLinkCase(twitter["content"])), remStop = True)


# In[ ]:


stem(remPunct(remUserLinkCase(twitter["content"])), remStop = False)


# In[ ]:


lem(remPunct(remUserLinkCase(twitter["content"])), remStop = True)


# In[ ]:


lem(remPunct(remUserLinkCase(twitter["content"])), remStop = False)

