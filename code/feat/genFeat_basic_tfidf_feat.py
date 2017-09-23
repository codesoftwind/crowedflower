import cPickle as pickle
from nlp_utils import tfidf_vec,bow_vec
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import vstack
from scipy.sparse.csr import csr_matrix

def groupData(dfData,key,additional_key=None):
    group_list = [key]
    if additional_key!=None:
        group_list.insert(0,additional_key)
    dfData['sample_index']=list(range(dfData.shape[0]))
    def groupFunc(df):
        return list(df['sample_index'])
    res = dict(dfData.groupby(group_list,as_index=True).apply(groupFunc))
    dfData.drop('sample_index',axis=1)
    return res

def cos_distance(x,y):
    return cosine_similarity(x,y)[0][0]

def generate_dist_stat_feat(dfTrain,idTrains,dfTest,idTest,group_dict,qids=None):
    sim = pairwise_distances(dfTest,dfTrain,metric='cosine')
    stat_feat = np.zeros((len(idTest),4*stats_feat_num),dtype=float)
    for i in range(len(idTest)):
        id_ = idTest[i]
        qid = None
        if qids!=None:
            qid = qids[i]
        for j in range(4):
            key_ = (qid,j+1) if qid!=None else (j+1)
            if key_ in group_dict:
                ind_list = [ind_ for ind_ in group_dict[key_] if idTrains[ind_]!=id_]
                sim_i = sim[i][ind_list]
                if len(sim_i)!=0:
                    series_sim = pd.Series(sim_i)
                    feat_quati = [series_sim.quantile(q) for q in quantiles_range]
                    feat_stat = [func(sim_i) for func in stats_func]
                    feat_temp = np.hstack((feat_quati,feat_stat))
                    stat_feat[i][j*stats_feat_num:(j+1)*stats_feat_num]=feat_temp
    return stat_feat



def extract_feat(path,vec_type,dfTrain,dfTest,feat_names,column_list):
    if vec_type=='tfidf':
        vec = tfidf_vec((1,3))
    elif vec_type=='bow':
        vec = bow_vec((1,3))
    vec.fit(dfTrain['all_text'])
    sampleGroupByRelevance = groupData(dfTrain, 'median_relevance')
    sampleGroupByQid = groupData(dfTrain, 'median_relevance', 'qid')

    for (feat_name,column) in zip(feat_names,column_list):
        #basic tfidf/bow vec
        dfTrain_vec = vec.transform(dfTrain[column])
        dfTest_vec = vec.transform(dfTest[column])
        with open('%strain_%s.pkl'%(path,feat_name),'wb') as f:
            pickle.dump(dfTrain_vec,f,-1)
        with open('%svalid_%s.pkl'%(path,feat_name),'wb') as f:
            pickle.dump(dfTest_vec,f,-1)

        # stastic feat of tfidf/bow vec
        if column in ['product_title','product_description'] :
            stat_relevance_feat = generate_dist_stat_feat(dfTrain_vec,dfTrain['id'].values,dfTrain_vec,dfTrain['id'].values,sampleGroupByRelevance)
            stat_qid_feat = generate_dist_stat_feat(dfTrain_vec,dfTrain['id'].values,dfTrain_vec,dfTrain['id'].values,sampleGroupByQid,dfTrain['qid'].values)

            with open('%strain_%s_%s_feat.pkl'%(path,feat_name,'cosine_stat_by_relevance'),'wb') as f:
                pickle.dump(stat_relevance_feat,f,-1)

            with open('%strain_%s_%s_feat.pkl'%(path,feat_name,'cosine_stat_by_qid_relevance'),'wb') as f:
                pickle.dump(stat_qid_feat,f,-1)

            stat_relevance_feat = generate_dist_stat_feat(dfTrain_vec, dfTrain['id'].values, dfTest_vec, dfTest['id'].values,
                                                          sampleGroupByRelevance)
            stat_qid_feat = generate_dist_stat_feat(dfTrain_vec, dfTrain['id'].values, dfTest_vec, dfTest['id'].values,
                                                    sampleGroupByQid, dfTest['qid'].values)
            with open('%svalid_%s_%s_feat.pkl' % (path, feat_name,'cosine_stat_by_relevance'), 'wb') as f:
                pickle.dump(stat_relevance_feat, f, -1)

            with open('%svalid_%s_%s_feat.pkl' % (path, feat_name,'cosine_stat_by_qid_relevance'), 'wb') as f:
                pickle.dump(stat_qid_feat, f, -1)


    for i in range(len(feat_names)):
        for j in range(len(feat_names)):
            if j>i:
                for mod in ['train','valid']:
                    with open('%s%s_%s.pkl' % (path,mod,feat_names[i]), 'rb') as f:
                        vec_i = pickle.load(f)
                    with open('%s%s_%s.pkl' % (path, mod,feat_names[j]), 'rb') as f:
                        vec_j = pickle.load(f)
                    cos_feat_i_j = np.array(map(cos_distance, vec_i, vec_j))[:, np.newaxis]
                    with open('%s%s_cosine_sim_%s_%s_feat.pkl' % (path,mod, feat_names[i], feat_names[j]), 'wb') as f:
                        pickle.dump(cos_feat_i_j, f, -1)

    # SVD feature
    for i,feat_name in enumerate(feat_names):
        with open('%strain_%s.pkl'%(path,feat_name),'rb') as f:
            x_train_vec = pickle.load(f)
        if i==0:
            x_train_all_vec = x_train_vec
        else:
            x_train_all_vec = vstack([x_train_all_vec,x_train_vec])
    for n_component in svd_n_components:
        SVD = TruncatedSVD(n_components=n_component,n_iter=5,algorithm='arpack')
        SVD.fit(x_train_all_vec)
        for (feat_name, column) in zip(feat_names, column_list):
            with open('%strain_%s.pkl' % (path, feat_name), 'rb') as f:
                train_vec = pickle.load(f)
            with open('%svalid_%s.pkl' % (path, feat_name), 'rb') as f:
                test_vec = pickle.load(f)
            dfTrain_svd_vec = csr_matrix(SVD.transform(train_vec))
            dfTest_svd_vec = csr_matrix(SVD.transform(test_vec))
            with open('%strain_%s_svd_%s.pkl' % (path, feat_name,n_component), 'wb') as f:
                pickle.dump(dfTrain_svd_vec, f, -1)
            with open('%svalid_%s_svd_%s.pkl' % (path, feat_name,n_component), 'wb') as f:
                pickle.dump(dfTest_svd_vec, f, -1)

            if column in ['product_title', 'product_description']:
                stat_relevance_feat = generate_dist_stat_feat(dfTrain_svd_vec, dfTrain['id'].values, dfTrain_svd_vec,
                                                              dfTrain['id'].values, sampleGroupByRelevance)
                stat_qid_feat = generate_dist_stat_feat(dfTrain_svd_vec, dfTrain['id'].values, dfTrain_svd_vec,
                                                        dfTrain['id'].values, sampleGroupByQid, dfTrain['qid'].values)

                with open('%strain_%s_svd_%s_%s_feat.pkl' % (path, feat_name,n_component,'cosine_stat_by_relevance'), 'wb') as f:
                    pickle.dump(stat_relevance_feat, f, -1)

                with open('%strain_%s_svd_%s_%s_feat.pkl' % (path,feat_name,n_component, 'cosine_stat_by_qid_relevance'), 'wb') as f:
                    pickle.dump(stat_qid_feat, f, -1)

                stat_relevance_feat = generate_dist_stat_feat(dfTrain_svd_vec, dfTrain['id'].values, dfTest_svd_vec,
                                                              dfTest['id'].values,
                                                              sampleGroupByRelevance)
                stat_qid_feat = generate_dist_stat_feat(dfTrain_svd_vec, dfTrain['id'].values, dfTest_svd_vec,
                                                        dfTest['id'].values,
                                                        sampleGroupByQid, dfTest['qid'].values)
                with open('%svalid_%s_svd_%s_%s_feat.pkl' % (path, feat_name,n_component,'cosine_stat_by_relevance'), 'wb') as f:
                    pickle.dump(stat_relevance_feat, f, -1)

                with open('%svalid_%s_svd_%s_%s_feat.pkl' % (path,feat_name,n_component, 'cosine_stat_by_qid_relevance'), 'wb') as f:
                    pickle.dump(stat_qid_feat, f, -1)
        for i in range(len(feat_names)):
            for j in range(len(feat_names)):
                if j > i:
                    for mod in ['train', 'valid']:
                        with open('%s%s_%s_svd_%s.pkl' % (path, mod, feat_names[i],n_component), 'rb') as f:
                            vec_i = pickle.load(f)
                        with open('%s%s_%s_svd_%s.pkl' % (path, mod, feat_names[j],n_component), 'rb') as f:
                            vec_j = pickle.load(f)
                        cos_feat_i_j = np.array(map(cos_distance, vec_i, vec_j))[:, np.newaxis]
                        with open('%s%s_svd_cosine_sim_%s_%s_feat.pkl' % (path, mod, feat_names[i], feat_names[j]),
                                  'wb') as f:
                            pickle.dump(cos_feat_i_j, f, -1)

        # individual svd feat
        for (feat_name, column) in zip(feat_names, column_list):
            with open('%strain_%s.pkl' % (path, feat_name), 'rb') as f:
                train_vec = pickle.load(f)
            with open('%svalid_%s.pkl' % (path, feat_name), 'rb') as f:
                test_vec = pickle.load(f)
            SVD = TruncatedSVD(n_components=n_component, n_iter=15)
            SVD.fit(train_vec)
            dfTrain_svd_vec = SVD.transform(train_vec)
            dfTest_svd_vec = SVD.transform(test_vec)
            with open('%strain_%s_individual_svd_%s.pkl' % (path, feat_name, n_component), 'wb') as f:
                pickle.dump(dfTrain_svd_vec, f, -1)
            with open('%svalid_%s_individual_svd_%s.pkl' % (path, feat_name, n_component), 'wb') as f:
                pickle.dump(dfTest_svd_vec, f, -1)

            if column in ['product_title', 'product_description']:
                stat_relevance_feat = generate_dist_stat_feat(dfTrain_svd_vec, dfTrain['id'].values, dfTrain_svd_vec,
                                                              dfTrain['id'].values, sampleGroupByRelevance)
                stat_qid_feat = generate_dist_stat_feat(dfTrain_svd_vec, dfTrain['id'].values, dfTrain_svd_vec,
                                                        dfTrain['id'].values, sampleGroupByQid, dfTrain['qid'].values)

                with open('%strain_%s_individual_svd_%s_%s_feat.pkl' % (path, feat_name, n_component, 'cosine_stat_by_relevance'),
                          'wb') as f:
                    pickle.dump(stat_relevance_feat, f, -1)

                with open('%strain_%s_individual_svd_%s_%s_feat.pkl' % (
                path, feat_name, n_component, 'cosine_stat_by_qid_relevance'), 'wb') as f:
                    pickle.dump(stat_qid_feat, f, -1)

                stat_relevance_feat = generate_dist_stat_feat(dfTrain_svd_vec, dfTrain['id'].values, dfTest_svd_vec,
                                                              dfTest['id'].values,
                                                              sampleGroupByRelevance)
                stat_qid_feat = generate_dist_stat_feat(dfTrain_svd_vec, dfTrain['id'].values, dfTest_svd_vec,
                                                        dfTest['id'].values,
                                                        sampleGroupByQid, dfTest['qid'].values)
                with open('%svalid_%s_individual_svd_%s_%s_feat.pkl' % (path, feat_name, n_component, 'cosine_stat_by_relevance'),
                          'wb') as f:
                    pickle.dump(stat_relevance_feat, f, -1)

                with open('%svalid_%s_individual_svd_%s_%s_feat.pkl' % (
                path, feat_name, n_component, 'cosine_stat_by_qid_relevance'), 'wb') as f:
                    pickle.dump(stat_qid_feat, f, -1)


if __name__=='__main__':
    with open('../../data/preprocessed_sampleTrainData.pkl','rb') as f:
        dfTrain = pickle.load(f)

    with open('../../data/preprocessed_sampleTestData.pkl','rb') as f:
        dfTest = pickle.load(f)

    with open('../../data/stratified_sample_relevance.pkl','rb') as f:
        skf = pickle.load(f)

    def concentrate_text(d):
        return ' '.join([d['query'],d['product_title'],d['product_description']])

    dfTrain['all_text'] = list(dfTrain.apply(concentrate_text,axis=1))
    dfTest['all_text'] = list(dfTest.apply(concentrate_text, axis=1))

    quantiles_range = np.arange(0, 1.5, 0.5)
    stats_func = [np.mean, np.std]
    stats_feat_num = len(quantiles_range) + len(stats_func)

    column_list = ["query", "product_title", "product_description"]
    vec_types = ['tfidf','bow']
    svd_n_components = [100, 150]
    run_times = 3

    for vec_type in vec_types:
        feat_names = ['%s_%s' % (x, vec_type) for x in column_list]
        # for run in range(run_times):
        #     for fold,(validIndex,trainIndex) in enumerate(skf[run]):
        #         path = '../../data/run%s_fold%s_'%(run+1,fold+1)
        #         extract_feat(path,vec_type,dfTrain.iloc[trainIndex].copy(),dfTrain.iloc[validIndex].copy(),feat_names,column_list)
        path = '../../data/all'
        extract_feat(path, vec_type, dfTrain, dfTest, feat_names, column_list)