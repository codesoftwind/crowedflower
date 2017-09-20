# -*- coding:utf-8 -*-

from nltk import word_tokenize

def getUnigram(sentence):
    wordList = word_tokenize(str(sentence).decode('utf-8'))
    assert type(wordList) == list  # 当assert为false的时候就会触发AssertionError异常
    return wordList

def getBigram(sentence, join_str):
    '''
    :param wordList: 分词之后的list
    :param join_str: 将词连接起来的连接符号
    :return: 返回2-gram的list
    '''
    wordList = word_tokenize(str(sentence).decode('utf-8'))
    assert type(wordList) == list
    bigramList = []
    wordListLen = len(wordList)
    if wordListLen >= 2:
        for i in range(0, wordListLen-1):
            bigramList.append(join_str.join([wordList[i], wordList[i+1]]))
    else:
        bigramList = getUnigram(wordList)
    return bigramList

def getTrigram(sentence, join_str):
    wordList = word_tokenize(str(sentence).decode('utf-8'))
    assert type(wordList) == list
    trigramList = []
    wordListLen = len(wordList)
    if wordListLen >= 3:
        for i in range(0, wordListLen-2):
            trigramList.append(join_str.join([wordList[i], wordList[i+1], wordList[i+2]]))
    else:
        trigramList = getBigram(wordList, join_str)
    return trigramList

