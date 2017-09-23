# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import cPickle
from copy import copy
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

## 下面三个函数是计算基本distance值的函数
def JaccardCoef(ngramList1, ngramList2):
    inter = set(ngramList1).intersection(set(ngramList2))
    unoin = set(ngramList1).union(set(ngramList2))
    coef = 0.0
    if len(unoin) != 0:
        coef = float(len(inter)) / len(unoin)
    return coef

def DiceDist(ngramList1, ngramList2):
    inter = set(ngramList1).intersection(set(ngramList2))
    dist = 0.0
    if len(ngramList1) + len(ngramList2) != 0:
        dist = float(2 * len(inter)) / (len(ngramList1) + len(ngramList2))
    return dist

def computeDistance(ngramList1, ngramList2, distKind):
    if distKind == 'JaccardCoef':
        distance = JaccardCoef(ngramList1, ngramList2)
    elif distKind == 'DiceDist':
        distance = DiceDist(ngramList1, ngramList2)
    return distance


# 提取基本的distance特征
# 这里没有返回df，但是新生成的feature已经存成文件了；返回如今的featNames
def extract_basic_distance_feat(path, mode, featNames, df):
    distances = ['JaccardCoef', 'DiceDist']
    ngrams = ['unigram', 'bigram', 'trigram']
    columns = ['query', 'title', 'description']
    for distance in distances:
        for ngram in ngrams:
            for i in range(0, len(columns)):
                for j in range(i+1, len(columns)):
                    featName1 = columns[i] + '_' + ngram
                    featName2 = columns[j] + '_' + ngram
                    newFeatName = '%s_of_%s_between_%s_and_%s'%(distance, ngram, columns[i], columns[j])
                    df[newFeatName] = df.apply(lambda x: computeDistance(x[featName1], x[featName2], distance), axis=1)
                    with open('%s/%s.%s.feat.pkl' % (path, mode, newFeatName), 'wb') as f:
                        cPickle.dump(df[newFeatName].values, f, -1)
                    df.drop(newFeatName, axis=1)
                    featNames.append(newFeatName)
    return featNames



# 按照给定的特征将数据集分组
def groupByKeys(df, keys):
    assert type(keys) == list  # keys必须是list，list中是用来分组的一个或多个key
    df['sampleIndex'] = range(df.shape[0])  # 暂时为df新增了一列，这一列范围是[0, shape[0]-1]，下面分组时，就是用这个来标识每一个样本
    tmp = df.groupby(keys, as_index=True).apply(lambda x: list(x['sampleIndex']))
    dic = dict(tmp)  # 这里的dic是一个字典，key是用来分组的key，value是一个list，list中是这个组中的样本的sampleIndex值
    df.drop('sampleIndex', axis=1)
    return dic

## 下面三个函数是计算统计distance值的函数
# 计算两个list中的两两元素之间的JaccardCoef距离，返回的是一个矩阵，这个函数会被pairwiseDistance调用
def pairwiseJaccardCoef(ngramList1, ngramList2):
    distanceMatrix = np.zeros((ngramList1.shape[0], ngramList2.shape[0]), dtype=float)
    for i in range(ngramList1.shape[0]):
        for j in range(ngramList2.shape[0]):
            distanceMatrix[i, j] = JaccardCoef(ngramList1[i], ngramList2[j])
    return distanceMatrix


# 计算两个list中的两两元素之间的DiceDist距离，返回的是一个矩阵，这个函数会被pairwiseDistance调用
def pairwiseDiceDist(ngramList1, ngramList2):
    distanceMatrix = np.zeros((ngramList1.shape[0], ngramList2.shape[0]), dtype=float)
    for i in range(ngramList1.shape[0]):
        for j in range(ngramList2.shape[0]):
            distanceMatrix[i, j] = DiceDist(ngramList1[i], ngramList2[j])
    return distanceMatrix


# 计算两个list中的两两元素之间的distance，返回的是一个矩阵
def pairwiseDistance(ngramList1, ngramList2, distKind):
    if distKind == 'JaccardCoef':
        distanceMatrix = pairwiseJaccardCoef(ngramList1, ngramList2)
    elif distKind == 'DiceDist':
        distanceMatrix = pairwiseDiceDist(ngramList1, ngramList2)
    return distanceMatrix


# 生成统计distance特征，这个函数会被extract_statistical_distance_feat调用
def generateStatsDistFeat(distance, groupDic, XTrain, trainIds, XTest, testIds, qids=None):
    funcs = [np.mean, np.std]  # mean和std这两个统计特征
    quantileRange = np.arange(0, 1.5, 0.5)  # 三个数量的统计特征

    statsFeatNum = len(funcs) + len(quantileRange)  # 这里是5个统计特征
    relevanceClass = 4  # relevance有1，2，3，4这四个取值
    statsFeats = 0 * np.ones((len(testIds), statsFeatNum*relevanceClass), dtype=float)

    # 计算XTrain中每一个元素与XTest中每一个元素的distance，所以会生成一个矩阵，然后，根据groupDic找出每一组的distance，然后算统计distance值
    distanceMatrix = pairwiseDistance(XTest, XTrain, distance)

    # 计算统计distance特征
    for i in range(len(testIds)):
        id = testIds[i]
        if qids is not None:  # 说明分组时的key是(qid, relevance)
            qid = qids[i]
        for j in range(relevanceClass):  # 每一个样本都要与每一个组进行计算
            key = (qid, j+1) if qids is not None else j+1
            if groupDic.has_key(key):
                groupIndexes = groupDic[key]  # 同一组的样本的index，这里的index是从0开始的连续值，而样本的id是从1开始的不连续的值
                indexes = [index for index in groupIndexes if id != trainIds[index]]  # trainIds[index]就得到了连续的index对应的不连续的id
                distances = distanceMatrix[i][indexes]  # 某一样本跟某一组中的每一个样本的distance值
                if len(distances) != 0:
                    feat = [func(distances)for func in funcs]  # 得到mean和std两个特征
                    distances = pd.Series(distances)
                    quantileFeat = distances.quantile(quantileRange)  # 得到三个数量的统计特征
                    feat = np.hstack((feat, quantileFeat))
                    statsFeats[i, j*statsFeatNum:(j+1)*statsFeatNum] = feat
    return statsFeats


# 提取统计distance特征，这里之所以叫统计特征，是因为特征其实是基本distance特征的一些统计值
def extract_statistical_distance_feat(path, dfTrain, dfTest, mode, featNames):
    # 新的特征list
    newFeatNames = copy(featNames)

    # 按照median_relevance分组；按照median_relevance和qid分组
    dicByRelevance = groupByKeys(dfTrain, ['median_relevance'])
    dicByRelevanceAndQid = groupByKeys(dfTrain, ['qid', 'median_relevance'])

    # 提取统计distance特征
    for distance in ['JaccardCoef', 'DiceDist']:  # 两种distance的计算方式
        for name in ['title', 'description']:  # 这里只计算title和description的统计distance特征，并且是title和title计算，description和description计算
            for ngram in ['unigram', 'bigram', 'trigram']:  # title和description均有三种ngram
                # train
                trainStatsDistFeatByRelevance = generateStatsDistFeat(distance, dicByRelevance, dfTrain[name+'_'+ngram].values, dfTrain['id'].values,
                                                                      dfTrain[name+'_'+ngram].values, dfTrain['id'].values)
                trainStatsDistFeatByRelevanceAndQid = generateStatsDistFeat(distance, dicByRelevanceAndQid, dfTrain[name+'_'+ngram].values, dfTrain['id'].values,
                                                                      dfTrain[name+'_'+ngram].values, dfTrain['id'].values, dfTrain['qid'].values)
                with open("%s/train.%s_%s_%s_stats_feat_by_relevance.feat.pkl" % (path, name, ngram, distance), "wb") as f:
                    cPickle.dump(trainStatsDistFeatByRelevance, f, -1)
                with open("%s/train.%s_%s_%s_stats_feat_by_query_relevance.feat.pkl" % (path, name, ngram, distance), "wb") as f:
                    cPickle.dump(trainStatsDistFeatByRelevanceAndQid, f, -1)
                # test，从这里可以看出，test数据集在提取统计distance特征时其实是每一个样本在跟train数据集中每一个组在计算，因为test根本没有relevance，而train和test是同分布的，所以这样计算依然有意义
                testStatsDistFeatByRelevance = generateStatsDistFeat(distance, dicByRelevance, dfTrain[name+'_'+ngram].values, dfTrain['id'].values,
                                                                     dfTest[name+'_'+ngram].values, dfTest['id'].values)
                testStatsDistFeatByRelevanceAndQid = generateStatsDistFeat(distance, dicByRelevanceAndQid, dfTrain[name+'_'+ngram].values, dfTrain['id'].values,
                                                                           dfTest[name+'_'+ngram].values, dfTest['id'].values, dfTest['qid'].values)
                with open("%s/%s.%s_%s_%s_stats_feat_by_relevance.feat.pkl" % (path, mode, name, ngram, distance), "wb") as f:
                    cPickle.dump(testStatsDistFeatByRelevance, f, -1)
                with open("%s/%s.%s_%s_%s_stats_feat_by_query_relevance.feat.pkl" % (path, mode, name, ngram, distance), "wb") as f:
                    cPickle.dump(testStatsDistFeatByRelevanceAndQid, f, -1)

                # 把新的特征名字添加进来
                newFeatNames.append('%s_%s_%s_stats_feat_by_relevance' % (name, ngram, distance))
                newFeatNames.append('%s_%s_%s_stats_feat_by_query_relevance' % (name, ngram, distance))

    return newFeatNames


# 将特征名存进特征名文件
def dumpFeatName(featNames, featNamesFilePath):
    with open(featNamesFilePath, "wb") as f:
        for i,feat_name in enumerate(featNames):
            if feat_name.startswith("count") or feat_name.startswith("pos_of"):
                f.write("('%s', SimpleTransform(config.count_feat_transform)),\n" % feat_name)
            else:
                f.write("('%s', SimpleTransform()),\n" % feat_name)

if __name__ == '__main__':

    # 加载数据
    print 'load the data'
    processedTrainDataPath = '../../data/preprocessedTrainData.pkl'
    processedTestDataPath = '../../data/preprocessedTestData.pkl'
    stratifiedKFoldPath = '../../data/stratified_query.pkl'
    with open(processedTrainDataPath, 'rb') as f:
        dfTrain = cPickle.load(f)
    with open(processedTestDataPath, 'rb') as f:
        dfTest = cPickle.load(f)
    with open(stratifiedKFoldPath, 'rb') as f:
        skf = cPickle.load(f)
    print '****************************'

    # 存特征名字的文件的路径和feat文件的根目录存特征名字的list
    featNamesFilePath = '../../data/distanceFeatName'
    featFolder = '../../data/feat'
    featNames = []

    # 是否生成statsFeature
    statsFeatFlag = True

    ## 生成特征

    # 为交叉验证数据生成特征
    print 'generate feat for cv'
    n_run = 3  # 每一次run的时候生成的交叉验证的数据集都不一样
    for run in range(n_run):
        for fold, (validIndex, trainIndex) in enumerate(skf[run]):
            print 'Run%s----Fold%s' % (run+1, fold+1)
            path = '%s/Run%s/Fold%s' % (featFolder, run+1, fold+1)

            # 生成basic distance feature
            featNames = extract_basic_distance_feat(path, 'train', featNames, dfTrain.iloc[trainIndex])
            featNames = extract_basic_distance_feat(path, 'valid', featNames, dfTrain.iloc[validIndex])

            # 生成statistical distance feature
            if statsFeatFlag:
                # 其实这里的统计distance特征也是分为train和valid，不过train没写
                featNames = extract_statistical_distance_feat(path, dfTrain.iloc[trainIndex], dfTrain.iloc[validIndex], 'valid', featNames)
    print '***********************************'


    # 为全体train和test生成特征
    print 'generate feat for all data'
    path = '%s/All' % featFolder
    extract_basic_distance_feat(path, 'train', copy(featNames), dfTrain)  # featNames.copy()是featNames的副本，对其的修改不会影响featNames
    # 其实这里的统计distance特征也是分为train和valid，不过train没写
    extract_statistical_distance_feat(path, dfTrain, dfTest, 'test', copy(featNames))
    print '**********************************'

    # 保存特征名
    dumpFeatName(featNames, featNamesFilePath)

    print 'Done!'
