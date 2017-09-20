import cPickle as pickle
from sklearn.preprocessing import LabelBinarizer
if __name__=='__main__':
    with open('../../data/preprocessedTrainData.pkl','rb') as f:
        dfTrain = pickle.load(f)

    with open('../../data/preprocessedTestData.pkl','rb') as f:
        dfTest = pickle.load(f)

    with open('../../data/stratified_relevance.pkl','rb') as f:
        skf = pickle.load(f)

    run_times = 3
    for run in range(run_times):
        for fold , (validIndex,trainIndex) in enumerate(skf[run]):
            lb = LabelBinarizer()
            fid_f_train = lb.fit_transform(dfTrain.loc[trainIndex]['qid'])
            fid_f_valid = lb.transform(dfTrain.loc[validIndex]['qid'])
            with open('../../data/run%s_fold%s_train_fid_feat.pkl'%(run+1,fold+1),'wb') as f:
                pickle.dump(fid_f_train,f,-1)
            with open('../../data/run%s_fold%s_valid_fid_feat.pkl'%(run+1,fold+1),'wb') as f:
                pickle.dump(fid_f_valid,f,-1)

    lb = LabelBinarizer()
    fid_all_train=lb.fit_transform(dfTrain['qid'])

    fid_all_test = lb.transform(dfTest['qid'])
    with open('../../data/all_train_fid_feat.pkl', 'wb') as f:
        pickle.dump(fid_all_train, f, -1)

    with open('../../data/all_test_fid_feat.pkl', 'wb') as f:
        pickle.dump(fid_all_test, f, -1)