#-*-coding:utf-8-*-

import pandas as pd
from bs4 import BeautifulSoup
from textblob import *
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

def read_csv(path):
    return pd.read_csv(path)

def drophtmltags(df):
    df['product_description'] = df['product_description'].apply(lambda x: BeautifulSoup(str(x), 'html5lib').get_text(separator=" "))
    return df

def spellcorrect(df):
    columns = ['query', 'product_title', 'product_description']
    for column in columns:
        df[column] = df[column].apply(lambda x: str(TextBlob(x).correct()))
    return df

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#word_tokenize将sentence分词，pos_tag识别词性，返回一个元组的list(word,pos)
def lemmatize_sentence(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return ' '.join(res)


def lemmatize(csv):
    columns = ['query', 'product_title', 'product_description']
    for column in columns:
        csv[column] = csv[column].apply(lambda x: lemmatize_sentence(str(x).decode('utf-8')))
    return csv


if __name__ == '__main__':
    df = read_csv('/Users/wangdong/repos/crowedflower/data/train_copy.csv').fillna("")
    df[['id', 'median_relevance', 'relevance_variance']] = df[['id', 'median_relevance', 'relevance_variance']].apply(pd.to_numeric)
    df = drophtmltags(df)
    df = spellcorrect(df)
    df = lemmatize(df)
    df.to_csv('newtarin.csv', encoding='utf-8')
    print df
