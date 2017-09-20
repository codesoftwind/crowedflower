import cPickle as pickle
from sklearn.cross_validation import StratifiedKFold
if __name__=='__main__':
    run_time = 3
    with open('../../data/preprocessedTrainData.pkl','rb') as f:
        dfTrain = pickle.load(f)
    skf = [0]*run_time
    for stratifiedLabel,key in zip(['relevance','query'],["median_relevance", "qid"]):
        for i in range(run_time):
            random_seed = 2017 + 100*(i+1)
            skf[i] = StratifiedKFold(dfTrain[key],n_folds=3,shuffle=True,random_state=random_seed)
        with open('../../data/stratified_%s.pkl'%(stratifiedLabel),'wb') as f:
            pickle.dump(skf,f,-1)