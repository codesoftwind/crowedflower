#-*-coding:utf-8-*-

"""
预处理
1.去除html标签
2.拼写改正
3.词性还原
4.同义词替换
5.去除停用词
"""


import pandas as pd
import cPickle
from bs4 import BeautifulSoup
from textblob import *
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import time
import sys
import numpy as np
import ngram
reload(sys)
sys.setdefaultencoding('utf-8')

def read_csv(path):
    return pd.read_csv(path)


def drophtmltags(df):
    df['product_description'] = df['product_description'].apply(lambda x: BeautifulSoup(str(x), 'html5lib').get_text(separator=" "))
    return df

def stemming(stemmer, sentence):
    stemmed = []
    wordList = word_tokenize(str(sentence).decode('utf-8'))
    for word in wordList:
        stemmedWord = stemmer.stem(word)
        stemmed.append(stemmedWord)
        if word != stemmedWord:
            print word, '---', stemmedWord
    return ' '.join(stemmed)

def stemDF(df):
    columns = ['query', 'product_title', 'product_description']
    stemmer = nltk.stem.SnowballStemmer('english')
    for column in columns:
        df[column] = df[column].apply(lambda x: stemming(stemmer, x))
    return df


# 生成qid，qid相同的样本的query也相同，经过处理发现大量的样本的query都相同
def generateQid(df):
    qid = {}
    for index, query in enumerate(np.unique(df['query']), start=1):
        qid[query] = index
    df['qid'] = map(lambda x: qid[x], df['query'])
    return df


# 为query，product_title，product_description生成n-gram
def generateNGram(df):
    # unigram
    df['query_unigram'] = df['query'].apply(lambda x: ngram.getUnigram(x))
    df['title_unigram'] = df['product_title'].apply(lambda x: ngram.getUnigram(x))
    df['description_unigram'] = df['product_description'].apply(lambda x: ngram.getUnigram(x))
    # bigram
    df['query_bigram'] = df['query'].apply(lambda x: ngram.getBigram(x, '_'))
    df['title_bigram'] = df['product_title'].apply(lambda x: ngram.getBigram(x, '_'))
    df['description_bigram'] = df['product_description'].apply(lambda x: ngram.getBigram(x, '_'))
    # trigram
    df['query_trigram'] = df['query'].apply(lambda x: ngram.getTrigram(x, '_'))
    df['title_trigram'] = df['product_title'].apply(lambda x: ngram.getTrigram(x, '_'))
    df['description_trigram'] = df['product_description'].apply(lambda x: ngram.getTrigram(x, '_'))
    return df


# 为test数据加上假的median_relevance和relevance_variance
def insertMRandRV(df):
    testNum = df.shape[0]
    df['median_relevance'] = np.ones(testNum)
    df['relevance_variance'] = np.zeros(testNum)
    return df

# def spellcorrect(df):
#     columns = ['query', 'product_title', 'product_description']
#     for column in columns:
#         df[column] = df[column].apply(lambda x: str(TextBlob(str(x).decode('utf-8')).correct()))
#     return df

if __name__ == '__main__':
    #(originDataPath, processedDataPath) = (sys.argv[1], sys.argv[2])
    print time.asctime(time.localtime(time.time()))

    # 处理train data
    originDataPath = '../../data/train.csv'
    df = read_csv(originDataPath).fillna("")
    #df[['id', 'median_relevance', 'relevance_variance']] = df[['id', 'median_relevance', 'relevance_variance']].apply(pd.to_numeric)
    df = drophtmltags(df)
    print '-------------------------------------'
    print 'drop html tags finished!'
    df = stemDF(df)
    print '-------------------------------------'
    print 'stemming finished!'
    df = generateQid(df)
    print '-------------------------------------'
    print 'generate qid finished!'
    df = generateNGram(df)
    print '-------------------------------------'
    print 'generate n-gram finished!'
    f = open('../../data/preprocessedTrainData.pkl', 'w+')
    cPickle.dump(df, f)
    f.close()
    
    # 处理test data
    originDataPath = '../../data/test.csv'
    df = read_csv(originDataPath)
    df = insertMRandRV(df)
    f = open('../../data/preprocessedTestData.pkl', 'w+')
    cPickle.dump(df, f)
    f.close()

    print time.asctime(time.localtime(time.time()))
