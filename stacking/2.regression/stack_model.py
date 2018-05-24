import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb

from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score

SEED = 1024

def xgb_r2_score(preds, dtrain):
	'''
	r2 evalution for example
	'''
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

def get_oof_regressor(clf, x_train, y_train, x_test, ntrain, ntest, NFOLDS = 5):
    kf = KFold(ntrain, n_folds = NFOLDS, random_state = SEED, shuffle = True)
    oof_train = np.zeros((ntrain, 1))
    oof_test = np.zeros((ntest, 1))
    oof_test_skf = np.empty((NFOLDS, ntest, 1))
    print("Anthor base regressor model")
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train[train_index]
        print("clf-----" + str(x_tr.shape[1]))
        x_te = x_train.iloc[test_index]
        clf.fit(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te).reshape(-1,1)
        print("--------->>>>>>>>:----r2 value :"+str(r2_score(y_train[test_index],oof_train[test_index]))) ## only to see the base model how well it perform
        oof_test_skf[i,:] = clf.predict(x_test).reshape(-1,1)
    oof_test[:] = oof_test_skf.mean(axis = 0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)



def get_oof_xgb_linear(parameters, x_train, y_train, x_test, ntrain, ntest, NFOLDS = 5):
    kf = KFold(ntrain, n_folds = NFOLDS, random_state = SEED, shuffle = True)
    oof_train = np.zeros((ntrain,1))
    oof_test = np.zeros((ntest,1))
    oof_test_skf = np.empty((NFOLDS,ntest,1))
    print("xgb-linear-base-model")
    for i,(train_index,test_index) in enumerate(kf):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train[train_index]
        y_te = y_train[test_index]
        print(x_tr.shape)
        x_te = x_train.iloc[test_index]
        dtrain = xgb.DMatrix(data = x_tr, label = y_tr)
        dval = xgb.DMatrix(data = x_te, label = y_te)
        dtest_train = xgb.DMatrix(data = x_te)
        dtest_test = xgb.DMatrix(data = x_test)
        watchlist = [(dtrain,'train'),(dval,'val')]
        bst = xgb.train(parameters,dtrain, num_boost_round = 100000, early_stopping_rounds = 100, evals = watchlist, feval= xgb_r2_score, maximize=True, verbose_eval=10)
        oof_train[test_index] = bst.predict(dtest_train).reshape(-1,1)
        print("--------->>>>>>>>:----r2 value :"+str(r2_score(y_train[test_index],oof_train[test_index])))
        oof_test_skf[i,:] = bst.predict(dtest_test).reshape(-1,1)

    oof_test[:] = oof_test_skf.mean(axis = 0)

    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)


def get_oof_lgb(parameters, x_train, y_train, x_test, ntrain, ntest, NFOLDS = 5):
    kf = KFold(ntrain, n_folds = NFOLDS, random_state = SEED, shuffle = True)
    oof_train = np.zeros((ntrain,1))
    oof_test = np.zeros((ntest,1))
    oof_test_skf = np.empty((NFOLDS,ntest,1))
    print("lgb_base_classier_model")
    for i,(train_index,test_index) in enumerate(kf):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train[train_index]
        y_te = y_train[test_index]
        print(x_tr.shape)

        x_te = x_train.iloc[test_index]
        dtrain = lgb.Dataset(x_tr, label = y_tr)
        dval = lgb.Dataset(x_te, label = y_te)
        dtest_train = x_te

        bst = lgb.train(parameters, dtrain, 3500000, valid_sets = dval, early_stopping_rounds = 100, verbose_eval=1000)

        oof_train[test_index] = bst.predict(dtest_train).reshape(-1,1)
        print("--------->>>>>>>>:----r2 value :"+str(r2_score(y_train[test_index],oof_train[test_index])))
        oof_test_skf[i,:] = bst.predict(x_test).reshape(-1,1)
        ## ---------------------------------batch end-------------------------------------
    oof_test[:] = oof_test_skf.mean(axis = 0)

    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)