import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

SEED = 2018

def get_oof_tree(clf, x_train, y_train, x_test, ntrain, ntest, NFOLDS = 5):
    kf = KFold(ntrain, n_folds = NFOLDS, random_state = SEED, shuffle = True)
    oof_train = np.zeros((ntrain, 3))
    oof_test = np.zeros((ntest, 3))
    oof_test_skf = np.empty((NFOLDS, ntest, 3))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train[train_index]
        print("clf-----" + str(x_tr.shape[1]))

        x_te = x_train.iloc[test_index]
        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict_proba(x_te)
        print("--------->>>>>>>>:----mlogloss Value :"+str(log_loss(y_train[test_index],oof_train[test_index])))

        oof_test_skf[i,:] = clf.predict_proba(x_test)

    oof_test[:] = oof_test_skf.mean(axis = 0)

    print(">>>>>>>>:----mlogloss Value :"+str(log_loss(y_train,oof_train)))
    return oof_train.reshape(-1,3),oof_test.reshape(-1,3)

## --------specify for the xgb model
def get_oof_xgb(parameters, x_train,y_train,x_test,ntrain,ntest, NFOLDS = 5):
    kf = KFold(ntrain, n_folds = NFOLDS, random_state = SEED, shuffle = True)
    oof_train = np.zeros((ntrain,3))
    oof_test = np.zeros((ntest,3))
    oof_test_skf = np.empty((NFOLDS,ntest,3))
    print("xgb_base_classier_model")

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
        ##-------------1106
        bst = xgb.train(parameters,dtrain, num_boost_round = 100000, early_stopping_rounds = 100, evals = watchlist,verbose_eval= 200)

        oof_train[test_index] = bst.predict(dtest_train)
        print("--------->>>>>>>>:----mlogloss Value :"+str(log_loss(y_train[test_index],oof_train[test_index])))
        oof_test_skf[i,:] = bst.predict(dtest_test)

    oof_test[:] = oof_test_skf.mean(axis = 0)

    return oof_train.reshape(-1,3),oof_test.reshape(-1,3)

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
        oof_test_skf[i,:] = clf.predict(x_test).reshape(-1,1)
    oof_test[:] = oof_test_skf.mean(axis = 0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)



def get_oof_xgb_linear(parameters, x_train,y_train,x_test,ntrain,ntest, NFOLDS = 5):
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
        ##----------1100------------
        bst = xgb.train(parameters,dtrain, num_boost_round = 100000, early_stopping_rounds = 100, evals = watchlist,verbose_eval= 30)

        oof_train[test_index] = bst.predict(dtest_train).reshape(-1,1)
        oof_test_skf[i,:] = bst.predict(dtest_test).reshape(-1,1)

    oof_test[:] = oof_test_skf.mean(axis = 0)

    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)
