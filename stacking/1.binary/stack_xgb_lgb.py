import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

# read dataset
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# gini function
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

def gini_lgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score, True

# define fold number
kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=42)
sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = np.zeros_like(test_id)

params_xgd = {
    'min_child_weight': 10.0,
    'objective': 'binary:logistic',
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.005,
    'gamma': 0.65,
    'num_boost_round' : 700
    }
params_lgb = {
    'max_depth': 7,
    'learning_rate': 0.005,
    'objective': 'binary'
}
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    
    d_train = lgb.Dataset(X_train, y_train)
    d_valid = lgb.Dataset(X_valid, y_valid)
    watchlist = [d_train, d_valid]

    model_lgb = lgb.train(params_lgb, d_train, 1600, watchlist, early_stopping_rounds = 70, feval = gini_lgb, verbose_eval = 100)

    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(test.values)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    model_xgb = xgb.train(params_xgd, d_train, 1600, watchlist, early_stopping_rounds = 70, feval = gini_xgb, maximize = True, verbose_eval = 100)

    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
    pred_xgb = model_xgb.predict(d_test, ntree_limit = mdl.best_ntree_limit)
    pred_lgb = model_lgb.predict(test.values)
	
	# 0.7 from xgb, 0.3 from lgb. You can play around here
    sub['target'] += (pred_xgb * 0.7 + pred_lgb * 0.3) / kfold