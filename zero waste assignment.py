# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 2020
Author: Brandi Beals
Description: Assignment 3
"""

######################################
# IMPORT PACKAGES
######################################

import os
#import json_lines
import pandas as pd
#import string
#import re
#import json
#import csv
from langdetect import detect
#import nltk
#from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from collections import Counter
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import scipy
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

######################################
# DIRECTORY OPERATIONS
######################################

# define function to list objects in a directory
def list_all(current_directory):
    for root, dirs, files in os.walk(current_directory):
        level = root.replace(current_directory, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

# set working directory and list items
os.chdir("C:/Users/bbeals/Dropbox (Personal)/Masters in Predictive Analytics/453-DL-56/Week 7/Beals B - Zero Waste/zerowaste")
cd = os.getcwd()
print(f'Your working directory is "{cd}"')
print('and has the following structure and files:')
list_all(cd)

# list available spiders/crawlers
os.system('scrapy list')

# make directory for storing complete html code for web page
page_dirname = 'html'
if not os.path.exists(page_dirname):
	os.makedirs(page_dirname)

######################################
# RUN CRAWLER
######################################

# output various formats: csv, JSON, XML, or jl for JSON lines
json_file = 'items.json'
if os.path.exists(json_file):
    os.remove(json_file)
#os.system(f'scrapy crawl zerowastecrawler -o {json_file}') # change .jl to .csv, .xml, or .json
os.system('scrapy crawl zerowaste -o {json_file}')
print(f'The Scrapy crawler has finished running.\nA {json_file} file has been created, containing its results.')

######################################
# IMPORT TEXT
######################################

# how many documents were gathered
corpus = pd.read_json(json_file)
print(f'{len(corpus)} documents were read into your corpus.')

# keep text longer than 300 characters
corpus['len'] = 0
for i in range(0, len(corpus)):
    corpus['len'].iloc[i] = len(corpus['body'].iloc[i]) 
corpus = corpus[corpus['len']>=300]

# keep text in English
corpus['lang'] = ''
for i in range(0, len(corpus)):
    corpus['lang'].iloc[i] = detect(corpus['body'].iloc[i]) 
corpus = corpus[corpus['lang']=='en']

######################################
# PREPROCESS TEXT
######################################

# non text elements
#re_punc = re.compile('[%s]' % re.escape(string.punctuation))
#re_print = re.compile('[^%s]' % re.escape(string.printable))
stop_words = set(stopwords.words('english'))

# create master lists
body = corpus['body'].tolist()
master_tokens = []
tokens = []
labels = []

# cycle through each document, clean text, and create master list of words
for i in range(0, len(corpus)):
    words = word_tokenize(corpus['body'].iloc[i])       # split by whitespace
    words = [w for w in words if w.isalpha()]           # remove punctuation
    words = [w for w in words if len(w) > 2]            # remove short words
    words = [w.lower() for w in words]                  # normalize case
    words = [w for w in words if not w in stop_words]   # remove stop words
    tokens.append(words)
    
    # add words to master list
    for w in words:
        if w not in master_tokens:
            master_tokens.append(w)
    
    # add url to label list
    labels.append(corpus['url'].iloc[i])

######################################
# VECTORIZE TEXT
######################################

### manual
# (1) counts
vec1 = []
for i in range(0, len(corpus)):
    count = Counter(tokens[i])
    vec1.append(dict(count))
vec1 = pd.DataFrame.from_dict(vec1)
vec1 = vec1.fillna(0)

# (2) calculate vectors
vec2 = vec1.copy()
vec2 = vec2.div(vec2.sum(axis=1), axis=0)
vec2_csr = scipy.sparse.csr_matrix(vec2.values)

# (3) scikit learn vectorize
#vec3 = []
#for i in range(0, len(corpus)):
#    count = CountVectorizer()
#    count.fit(tokens[i])
#    vector = count.transform(tokens[i])
#    vec3.append(vector.toarray())

### TF-IDF
# (4) keras
tokenize = Tokenizer()
tokenize.fit_on_texts(body)
vec4 = tokenize.texts_to_matrix(body, mode='tfidf')
vec4 = pd.DataFrame(vec4)
del vec4[0]

# create list of tokens to use as column names
cols = []
for k in tokenize.word_index:
    cols.append(k)
vec4.columns = cols

vec4 = vec4.reindex(sorted(vec4.columns), axis=1)

#print(tokenize.word_counts)
#print(tokenize.document_count)
#print(tokenize.word_index)
#print(tokenize.word_docs)

# (5) scikit learn
vectorize = TfidfVectorizer()
vec5_csr = vectorize.fit_transform(body)
vec5 = pd.DataFrame(vec5_csr.toarray(), columns=vectorize.get_feature_names())

#print(vectorize.get_feature_names())
#print(vec5.shape)

### word embeddings
# (6) Doc2Vec 100
tagged = [TaggedDocument(doc, [i]) for i, doc in enumerate(body)]
doc_vec = Doc2Vec(tagged)

vec6 = pd.DataFrame()
for i in range(0, len(corpus)):
    vector = pd.DataFrame(doc_vec.infer_vector(tokens[i])).transpose()
    vec6 = vec6.append(vector)

vec6 = vec6.reset_index()
del vec6['index']

vec6_csr = scipy.sparse.csr_matrix(vec6.values)

# (7) Doc2Vec 1000
tagged = [TaggedDocument(doc, [i]) for i, doc in enumerate(body)]
doc_vec2 = Doc2Vec(tagged, vector_size=1000)

vec7 = pd.DataFrame()
for i in range(0, len(corpus)):
    vector = pd.DataFrame(doc_vec2.infer_vector(tokens[i])).transpose()
    vec7 = vec7.append(vector)

vec7 = vec7.reset_index()
del vec7['index']

vec7_csr = scipy.sparse.csr_matrix(vec7.values)

######################################
# COMPARISON OF VECTORS
######################################



######################################
# CLUSTER ANALYSIS
######################################

k = 8 # number of clusters
random_seed = 9999 # random set of initial values for reproducibility

# k-means cluster on manual vector
km1 = KMeans(n_clusters=k, random_state=random_seed)
km1.fit(vec2_csr)
km1_clusters = km1.labels_.tolist()

# k-means cluster on tf-idf vector
km2 = KMeans(n_clusters=k, random_state=random_seed)
km2.fit(vec5_csr)
km2_clusters = km2.labels_.tolist()

# k-means cluster on word embeddings vector
km3 = KMeans(n_clusters=k, random_state=random_seed)
km3.fit(vec6_csr)
km3_clusters = km3.labels_.tolist()

# k-means cluster on word embeddings vector 1000
km4 = KMeans(n_clusters=k, random_state=random_seed)
km4.fit(vec7_csr)
km4_clusters = km4.labels_.tolist()

# output results
km_df = corpus.copy()
km_df['manual vec clusters'] = km1_clusters
km_df['tf-idf vec clusters'] = km2_clusters
km_df['embeddings vec clusters'] = km3_clusters
km_df['embeddings vec clusters 1000'] = km4_clusters

km_df.to_excel('clusters.xlsx')
# https://public.tableau.com/profile/brandi.beals#!/vizhome/ZeroWasteAssignment3/ClusterAnalysis

# common words for each cluster
common_words1 = km1.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words1):
    print(str(num) + ' : ' + ', '.join(list(vec2.columns)[word] for word in centroid))

common_words2 = km2.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words2):
    print(str(num) + ' : ' + ', '.join(vectorize.get_feature_names()[word] for word in centroid))

######################################
# MULTIDIMENSIONAL SCALING
######################################

n = 2 # number of components
random_seed = 9999 # random set of initial values for reproducibility

# t-sne on manual vector
tsne1 = TSNE(n_components=n,  metric="euclidean", random_state=random_seed)
tsne1_fit = tsne1.fit_transform(vec2_csr.toarray())

# t-sne on tf-idf vector
tsne2 = TSNE(n_components=n,  metric="euclidean", random_state=random_seed)
tsne2_fit = tsne2.fit_transform(vec5_csr.toarray())

# t-sne on word embeddings vector
tsne3 = TSNE(n_components=n,  metric="euclidean", random_state=random_seed)
tsne3_fit = tsne3.fit_transform(vec6_csr.toarray())

# output results
tsne_df = km_df.copy()
tsne_df['manual vec scaling 1'] = tsne1_fit[:,0]
tsne_df['manual vec scaling 2'] = tsne1_fit[:,1]
tsne_df['tf-idf vec scaling 1'] = tsne2_fit[:,0]
tsne_df['tf-idf vec scaling 2'] = tsne2_fit[:,1]
tsne_df['embeddings vec scaling 1'] = tsne3_fit[:,0]
tsne_df['embeddings vec scaling 2'] = tsne3_fit[:,1]

tsne_df.to_excel('clusters and tsne.xlsx')
# link to Tableau dashboard analysis

######################################
# CLUSTER ANALYSIS - TERMS AS OBJECTS
######################################

k = 8 # number of clusters
random_seed = 9999 # random set of initial values for reproducibility

# k-means cluster on manual vector
km1t = KMeans(n_clusters=k, random_state=random_seed)
km1t.fit(vec2_csr.transpose())
km1t_clusters = km1t.labels_.tolist()

# output results
km1t_df = pd.DataFrame(km1t_clusters, index=list(vec2.columns))
km1t_df.to_excel('term clusters manual.xlsx')

# k-means cluster on tf-idf vector
km2t = KMeans(n_clusters=k, random_state=random_seed)
km2t.fit(vec5_csr.transpose())
km2t_clusters = km2t.labels_.tolist()

# output results
km2t_df = pd.DataFrame(km2t_clusters, index=vectorize.get_feature_names())
km2t_df.to_excel('term clusters tfidf.xlsx')

######################################
# MULTIDIMENSIONAL SCALING - TERMS AS OBJECTS
######################################

n = 2 # number of components
random_seed = 9999 # random set of initial values for reproducibility

# t-sne on manual vector
tsne1t = TSNE(n_components=n,  metric="euclidean", random_state=random_seed)
tsne1t_fit = tsne1t.fit_transform(vec2_csr.transpose().toarray())

# output results
tsne1t_df = pd.DataFrame(tsne1t_fit, index=vec2.columns)
tsne1t_df.to_excel('term tsne manual.xlsx')

# t-sne on tf-idf vector
tsne2t = TSNE(n_components=n,  metric="euclidean", random_state=random_seed)
tsne2t_fit = tsne2t.fit_transform(vec5_csr.transpose().toarray())

# output results
tsne2t_df = pd.DataFrame(tsne2t_fit, index=vectorize.get_feature_names())
tsne2t_df.to_excel('term tsne tfidf.xlsx')







































######################################
# FIND EQUIVALENT CLASSES
######################################



######################################
# IDENTIFY TOPICS
######################################

### manually: which token is the most common
top_topic = pd.DataFrame()

for i in range(0, len(corpus)):
    if len(manual_vec['Count'][i])>0:
        top_topic = top_topic.append({
                'Code':corpus['Code'][i],
                'Topic':manual_vec['Count'][i].most_common(1)[0][0]
                }, ignore_index=True)

### LDA classifier

### PCA

### CNN



