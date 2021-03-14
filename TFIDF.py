
from bs4 import BeautifulSoup, Comment
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

from os import listdir
from os.path import isfile, join

import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math


with open('./tf_idf.txt', encoding='utf8') as f:
    TF_IDF = eval(f.read())


def convert_lower_case(data):
    return np.char.lower(data)


def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data


def doc_freq(word, DF):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


def get_content_title(file_name):

    with open(file_name, 'r') as f:
        data = f.read()

    soup = BeautifulSoup(data, 'html.parser')
    text = soup.find_all(text=True)
    try:
        title = soup.find('title').text
    except:
        print('No title found')
        title = ''

    output = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        # there may be more elements you don't want, such as "style", etc.
    ]

    for t in text:
        if t.parent.name not in blacklist:
            if isinstance(t, Comment):
                continue
            output += '{} '.format(t)

    return output, title


def run(mypath='./resources/HTML/'):
    dataset = []

    c = False

    # folders = [x[0] for x in os.walk('./resources/HTML/')]
    # folders[0] = folders[0][:len(folders[0]) - 1]

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # for i in onlyfiles:
    #     file = open(mypath+i, 'r')
    #     text = file.read().strip()
    #     file.close()
    #
    #     file_name = re.findall('><A HREF="(.*)">', text)
    #     file_title = re.findall('<BR><TD> (.*)\n', text)
    #
    #     if c == False:
    #         file_name = file_name[2:]
    #         c = True
    #
    #     print(len(file_name), len(file_title))
    #
    #     for j in range(len(file_name)):
    #         dataset.append((str(i) + "/" + str(file_name[j]), file_title[j]))

    # N = len(dataset)

    alpha = 0.3
    processed_text = []
    processed_title = []

    for i in onlyfiles:
        content, title = get_content_title(mypath+i)

        processed_text.append(word_tokenize(str(preprocess(content.strip()))))
        processed_title.append(word_tokenize(str(preprocess(title.strip()))))

    N = len(processed_text)


    # calculating DF for all words
    DF = {}
    for i in range(N):
        tokens = processed_text[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}

    for i in DF:
        DF[i] = len(DF[i])

    doc = 0

    tf_idf = {}

    for i in range(N):

        tokens = processed_text[i]

        counter = Counter(tokens)
        words_count = len(tokens)

        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = doc_freq(token, DF)
            idf = np.log((N + 1) / (df + 1))

            tf_idf[doc, token] = tf * idf

        doc += 1

    for i in tf_idf:
        tf_idf[i] *= alpha


def matching_score(k, query, tf_idf):
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))

    # print("Matching Score")
    # print("\nQuery:", query)
    # print("")
    # print(tokens)

    query_weights = {}

    for key in tf_idf:

        if key[1] in tokens:
            try:
                query_weights[key[0]]['value'] += tf_idf[key]
                query_weights[key[0]]['word'].append(key[1])
            except:
                query_weights[key[0]] = {'value': tf_idf[key], 'word': [key[1]]}

    query_weights = sorted(query_weights.items(), key=lambda x: x[1]['value'], reverse=True)

    # print("")
    #
    # l = []
    #
    # for i in query_weights[:k]:
    #     l.append(i[1])
    #
    # print(l)
    result = []
    for q in query_weights:
        if len(result) == 5:
            break
        for word in q[1]['word']:
            if word not in result:
                result.append(word)
                if len(result) == 5:
                    break

    return result

def tf_idf_library(document):
    """
    this function calculates the tf_idf of given documents.
    document: dataset with the format ["today is a good day", "machine learning and cyber security"]
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfIdfVectorizer=TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(document)
    df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
if __name__ == '__main__':
    run()
    matching_score(5, "paypal, please login with your email and password credit card number", TF_IDF)