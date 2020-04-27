# -*- coding: utf-8 -*-
"""
Author: Brandi Beals
Description: Assignment 1
"""

# import packages
from bs4 import BeautifulSoup
import requests
#import lxml.html
import string
import re
import nltk
import pandas as pd
import os

# directory operations
def list_all(current_directory):
    for root, dirs, files in os.walk(current_directory):
        level = root.replace(current_directory, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

os.chdir("C:/Users/bbeals/Dropbox (Personal)/Masters in Predictive Analytics/453-DL-56/Week 3/Beals B - Zero Waste/zerowaste")
cd = os.getcwd()
list_all(cd)
os.system('scrapy list')

# make directory for storing complete html code for web page
page_dirname = 'html'
if not os.path.exists(page_dirname):
	os.makedirs(page_dirname)

# output various formats: csv, JSON, XML, or jl for JSON lines
json_file = 'items.jl'
if os.path.exists(json_file):
    os.remove(json_file)
os.system('scrapy crawl zerowaste -o items.jl') # change .jl to .csv, .xml, or .json



## starter website urls
#urls = ['https://en.wikipedia.org/wiki/Zero_waste',
#            'https://mashable.com/article/zero-waste/',
#            'https://www.glasspantrymilwaukee.com/zerowaste/',
#            'https://trashisfortossers.com/compost-lets-break-it-down-literally/',
#            'https://www.goingzerowaste.com/blog-posts-for-beginners',
#            'https://www.reddit.com/r/ZeroWaste/',
#            'https://zerowastehome.com/blog/',
#            'https://zerowaste.com/']
#
## webscrape patagonia story
#url = 'https://www.patagonia.com/stories/the-trees-do-better-standing-up/story-79532.html'
#webpage = requests.get(url)
#
#print(webpage.status_code)  # status should be 200
#print(webpage.encoding)     # encoding should be utf8
#html = webpage.text         # returns HTML
#
##html = lxml.html.fromstring(html)
#
#soup = BeautifulSoup(html)
#print(type(soup))
##print(soup.prettify())
#print(soup.title.text)
#
##[x.extract() for x in soup.find_all('script')]
#
#content = [x.text for x in soup.find_all('p')]
#print(len(content))
#print(type(content))
#
#text = ''.join(content)
#
## tokenization
#words = text.split()
#
## remove punctuation
#re_punc = re.compile('[%s]' % re.escape(string.punctuation))
#stripped = [re_punc.sub('', w) for w in words]
#
## normalize case
#tokens = [word.lower() for word in stripped]
#print(tokens[:100])
#
## create word frequency list
#wordfreq = [tokens.count(p) for p in tokens]
#dict = dict(list(zip(tokens,wordfreq)))
#
## convert to dataframe
#df = pd.DataFrame(dict.items())
#
## stop words
## stemming


