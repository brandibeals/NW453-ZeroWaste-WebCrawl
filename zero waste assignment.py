# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 2020
Author: Brandi Beals
Description: Assignment 4
"""

######################################
# IMPORT PACKAGES
######################################

import os
import pandas as pd
from langdetect import detect
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from random import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping

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
os.chdir("C:/Users/bbeals/Dropbox (Personal)/Masters in Predictive Analytics/453-DL-56/Week 10/NW453-ZeroWaste-WebCrawl")
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

body_clean = []
for i in range(0, len(tokens)):
    body_clean.append(" ".join(tokens[i]))

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

vectorize2 = TfidfVectorizer()
vec51_csr = vectorize2.fit_transform(body_clean)
vec51 = pd.DataFrame(vec51_csr.toarray(), columns=vectorize2.get_feature_names())

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

# (8) Doc2Vec 1000 Clean
tagged_clean = [TaggedDocument(doc, [i]) for i, doc in enumerate(body_clean)]
doc_vec3 = Doc2Vec(tagged_clean, vector_size=1000)

vec8 = pd.DataFrame()
for i in range(0, len(corpus)):
    vector = pd.DataFrame(doc_vec3.infer_vector(tokens[i])).transpose()
    vec8 = vec8.append(vector)

vec8 = vec8.reset_index()
del vec8['index']

vec8_csr = scipy.sparse.csr_matrix(vec8.values)

######################################
# COMPARISON OF VECTORS
######################################

topn = 10

v1 = vec1.sum(axis=0)
v1 = v1.sort_values(ascending=False)
sns.barplot(x=v1[:topn], y=v1[:topn].index, color='lightgray').set_title('Top 10 Words in Vector 1\n(Counts of Cleaned Text)')

v2 = vec2.sum(axis=0)
v2 = v2.sort_values(ascending=False)
sns.barplot(x=v2[:topn], y=v2[:topn].index, color='lightgray').set_title('Top 10 Words in Vector 2\n(Rate of Use in Cleaned Text)')

v4 = vec4.sum(axis=0)
v4 = v4.sort_values(ascending=False)
sns.barplot(x=v4[:topn], y=v4[:topn].index, color='lightgray').set_title('Top 10 Words in Vector 4\n(Keras TF-IDF of Raw Text)')

v5 = vec5.sum(axis=0)
v5 = v5.sort_values(ascending=False)
sns.barplot(x=v5[:topn], y=v5[:topn].index, color='lightgray').set_title('Top 10 Words in Vector 5\n(Scikit Learn TF-IDF of Raw Text)')

v51 = vec51.sum(axis=0)
v51 = v51.sort_values(ascending=False)
sns.barplot(x=v51[:topn], y=v51[:topn].index, color='lightgray').set_title('Top 10 Words in Vector 5\n(Scikit Learn TF-IDF of Cleaned Text)')

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

# k-means cluster on tf-idf vector cleaned
km21 = KMeans(n_clusters=k, random_state=random_seed)
km21.fit(vec51_csr)
km21_clusters = km21.labels_.tolist()

# k-means cluster on word embeddings vector
km3 = KMeans(n_clusters=k, random_state=random_seed)
km3.fit(vec6_csr)
km3_clusters = km3.labels_.tolist()

# k-means cluster on word embeddings vector 1000
km4 = KMeans(n_clusters=k, random_state=random_seed)
km4.fit(vec7_csr)
km4_clusters = km4.labels_.tolist()

# k-means cluster on word embeddings vector 1000 cleaned
km5 = KMeans(n_clusters=k, random_state=random_seed)
km5.fit(vec8_csr)
km5_clusters = km5.labels_.tolist()

# output results
km_df = corpus.copy()
km_df['manual vec clusters'] = km1_clusters
km_df['tf-idf vec clusters'] = km2_clusters
km_df['tf-idf vec clusters cleaned'] = km21_clusters
km_df['embeddings vec clusters'] = km3_clusters
km_df['embeddings vec clusters 1000'] = km4_clusters
km_df['embeddings vec clusters 1000 cleaned'] = km5_clusters

km_df.to_excel('clusters.xlsx')
# https://public.tableau.com/profile/brandi.beals#!/vizhome/ZeroWasteAssignment3/ClusterAnalysis

# common words for each cluster
common_words1 = km1.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words1):
    print(str(num) + ' : ' + ', '.join(list(vec2.columns)[word] for word in centroid))

common_words2 = km2.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words2):
    print(str(num) + ' : ' + ', '.join(vectorize.get_feature_names()[word] for word in centroid))

common_words21 = km21.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words21):
    print(str(num) + ' : ' + ', '.join(vectorize2.get_feature_names()[word] for word in centroid))

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
# NEURAL NETWORK
######################################

# followed exercise documented here: 
# https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb

### data preparation
# split documents into sentences
sentences = []

for i in range(0, len(corpus)):
    sent = sent_tokenize(corpus['body'].iloc[i])        # split by sentence
    for s in sent:
        sentences.append(s)

# cycle through each sentence and normalize text
sentence_tokens = []

for i in range(0, len(sentences)):
    words = word_tokenize(sentences[i])                 # split by whitespace
    words = [w for w in words if w.isalpha()]           # remove punctuation
    words = [w.lower() for w in words]                  # normalize case
    sentence_tokens.append(words)

# specify any words to be ignored
ignored_words = set()

# cut the text into semi-redundant sequences
step = 1
sequence_len = 5
first_words = []
next_words = []
ignored = 0

for i in range(0, len(sentence_tokens)):
    s = sentence_tokens[i]
    for w in range(0, len(s) - sequence_len, step):
        first_words.append(s[w: w + sequence_len])
        next_words.append(s[w + sequence_len])
        # Only add sequences where no word is in ignored_words
#        if len(set(body_tokens[i: i+sequence_len+1]).intersection(ignored_words)) == 0:
#            first_words.append(body_tokens[i: i + sequence_len])
#            next_words.append(body_tokens[i + sequence_len])
#        else:
#            ignored = ignored+1

print('Ignored sequences:', ignored)
print('Remaining sequences:', len(first_words))

# shuffle data
map_position = list(zip(first_words, next_words))
shuffle(map_position)
first_words, next_words = zip(*map_position)

# split data
x_train,x_test,y_train,y_test = train_test_split(first_words, next_words, test_size=0.3)

### try various model structures

random_seed = 9999 # random set of initial values for reproducibility

# set up base class for callbacks to monitor training
# and for early stopping during training
tf.keras.callbacks.Callback()

# (1) RNN
model1 = Sequential()
model1.add(Bidirectional(LSTM(128), input_shape=(sequence_len, len(x_train))))
#model1.add(Dropout(dropout))
model1.add(Dense(len(x_train)))
model1.add(Activation('softmax'))












# (2) LSTM



#densly connected neural net

#1-dimensional CNN

# (3) 

# which one performed best

### try different hyperparameters
# early stopping, regularization, etc
# try cutting out uncommon words

# (1)

# (2)

# (3)

# which one performed best

### evaluate on various metrics (ROC curve for binary, F1 for multinomial classification)


### visualize accuracy and loss plots



### LDA classifier

### PCA

### CNN



