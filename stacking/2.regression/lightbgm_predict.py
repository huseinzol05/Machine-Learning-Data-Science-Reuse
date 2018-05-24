import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb

params = {}
params['learning_rate'] = 0.01
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'rmse'
params['sub_feature'] = 0.5
params['num_leaves'] = 100
params['min_data'] = 1000
params['min_hessian'] = 5
params['huber_delta'] = 0.5
params['lambda_l2'] = 0.0005

y_test = np.zeros(len(X_test))
fold = 5

for i, (train_ind, val_ind) in enumerate(KFold(n_splits = fold, shuffle = True, random_state = 1989).split(X_train)):
    print('Training model #%d' % i)
    d_train = lgb.Dataset(X_train[train_ind], label = y_train[train_ind])
    d_valid = lgb.Dataset(X_train[val_ind], y_train[val_ind])
    watchlist = [d_valid]
    clf = lgb.train(params, d_train, 5000, watchlist)
    y_test += clf.predict(X_test)
    
y_test /= fold