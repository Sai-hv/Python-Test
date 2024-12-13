
##test1
!pip install polyfuzz
!pip install google-search-results
!pip install plotly
!pip install gensim

#Libraries for data manipulation
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import plotly.express as px

#Libraries fo preprocessing/tokenization
from gensim.parsing.preprocessing import remove_stopwords
import string
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

#Libraries for vectorisation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#Libraries for similarity clustering
from sklearn.cluster import KMeans
from polyfuzz.models import TFIDF
from polyfuzz import PolyFuzz

from serpapi import GoogleSearch

serp_apikey = "" 

params = {
    "engine": "google",
    "q": "latte",
    "location": "United Kingdom",
    "google_domain": "google.com",
    "gl": "uk",
    "hl": "en",
    "num": 10,
    "api_key": serp_apikey
}

client = GoogleSearch(params)
data = client.get_dict()

# access "organic results"
df = pd.DataFrame(data['organic_results'])
df.to_csv('results_1.csv', index=False)
df

SERP_One = pd.read_csv('/content/results_1.csv')
df = pd.DataFrame(SERP_One, columns=['link'])
df.columns = ['URL1']
df

df ['snippet_highlighted_words'] =  df ['Highlighted_Words1'].str.replace("\[|\"|\]|\'", "")
df = df.fillna(0)
df.isnull().sum()
df

SERP_One = pd.read_csv('/content/results_1.csv')
# Assuming 'Highlighted_Words1' is a column in 'SERP_One'
# Include it while creating df
df = pd.DataFrame(SERP_One, columns=['link', 'snippet_highlighted_words', 'snippet']) # Added 'Highlighted_Words1' column selection
df.columns = ['URL1', 'Highlighted_Words1', 'snippet1'] # Renamed columns

df = df.fillna(0)
df.isnull().sum()
df

import nltk
textlist = df['snippet'].to_list()

from collections import Counter
x = Counter(textlist)

#download stopwords list to remove what is not needed
nltk.download('stopwords')
from nltk.corpus import stopwords
stoplist = stopwords.words('english')

#create dataframe with bigrams and trigrams
from sklearn.feature_extraction.text import CountVectorizer
c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3)) #can also select bigrams only
# matrix of ngrams
ngrams = c_vec.fit_transform(df['snippet'])
# count frequency of ngrams
count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})
            
#Get the output
df_ngram.head(20).style.background_gradient()

from serpapi import GoogleSearch

serp_apikey = "a80a9929e18f27cd6b9bc4a55c70200aa6303f84acbbfbe9cbcaa69733c6a57f" 

params = {
    "engine": "google",
    "q": "cappuccino",
    "location": "United Kingdom",
    "google_domain": "google.com",
    "gl": "uk",
    "hl": "en",
    "num": 10,
    "api_key": serp_apikey
}

client = GoogleSearch(params)
data = client.get_dict()

# access "organic results"
df2 = pd.DataFrame(data['organic_results'])
df2.to_csv('results_2.csv', index=False)
df2

SERP_Two = pd.read_csv('/content/results_2.csv')
df2 = pd.DataFrame(SERP_Two, columns=['link', 'snippet'])
df2.columns = ['URL2', 'snippet2']
df2


import nltk
textlist = df2['snippet'].to_list()

from collections import Counter
x = Counter(textlist)

#download stopwords list to remove what is not needed
nltk.download('stopwords')
from nltk.corpus import stopwords
stoplist = stopwords.words('english')

#create dataframe with bigrams and trigrams
from sklearn.feature_extraction.text import CountVectorizer
c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3)) #can also select bigrams only
# matrix of ngrams
ngrams = c_vec.fit_transform(df2['snippet'])
# count frequency of ngrams
count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})
            
#Get the output
df_ngram.head(20).style.background_gradient()

Comparison = pd.concat([df, df2], axis=1)
Comparison = Comparison.dropna()
Comparison

URL1 = Comparison['URL1'].tolist()
cleanedList2 = [x for x in URL1 if str(x) != 'nan']
len(URL1)

URL2 = Comparison['URL1'].tolist()
cleanedList2 = [x for x in URL2 if str(x) != 'nan']
len(URL2)

results_1 = df['snippet1'].tolist()
results_2 = df2['snippet2'].tolist()

tfidf = TFIDF(n_gram_range=(3,3), min_similarity=0.95, cosine_method='knn')
model = PolyFuzz(tfidf)
model.match(results_1, results_2)
similarity = model.get_matches()

outcome = pd.DataFrame(similarity)
outcome.sort_values('Similarity', ascending=False, inplace=True)
outcome
