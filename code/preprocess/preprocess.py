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
from bs4 import BeautifulSoup
from textblob import *
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import time
import sys
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

# def spellcorrect(df):
#     columns = ['query', 'product_title', 'product_description']
#     for column in columns:
#         df[column] = df[column].apply(lambda x: str(TextBlob(str(x).decode('utf-8')).correct()))
#     return df

if __name__ == '__main__':
    (originDataPath, processedDataPath) = (sys.argv[1], sys.argv[2])
    print time.asctime(time.localtime(time.time()))
    df = read_csv(originDataPath).fillna("")
    #df[['id', 'median_relevance', 'relevance_variance']] = df[['id', 'median_relevance', 'relevance_variance']].apply(pd.to_numeric)
    df = drophtmltags(df)
    print '-------------------------------------'
    print 'drop html tags finished!'
    df = stemDF(df)
    print '-------------------------------------'
    print 'stemming finished!'
    df.to_csv(processedDataPath, encoding='utf-8')
    print time.asctime(time.localtime(time.time()))
