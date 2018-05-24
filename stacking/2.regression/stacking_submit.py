import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score

pd.options.mode.chained_assignment = None 

# read datasets
train = pd.read_csv('../../input/train.csv')
test = pd.read_csv('../../input/test.csv')

test_id = test.ID
y_train = train.y

#----------------read the base model prediction files-------------------
x_train = pd.read_csv("../stack_data/....") # define yourself
x_test = pd.read_csv("../stack_data/....") 

name_col = ["col" + str(i) for i in range(x_train.shape[1])]
x_test.columns = name_col
x_train.columns = name_col

dtrain = xgb.DMatrix(data = x_train, label = y_train)
params = {
        'objective': 'reg:linear',
        'metric': 'rmse',
        'max_depth' : 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'learning_rate': 0.001,
        'seed': 0,
        'nthread': -1,
        'silent':True,
        'verbose':0
    }

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

clf = xgb.cv(params, dtrain, 10000, early_stopping_rounds = 50, feval= xgb_r2_score, maximize=True, verbose_eval=100)

best_rounds = np.argmax(clf['test-r2-mean'])
print("----------------------------------------------------")
print(clf.iloc[best_rounds])
files_name = clf.iloc[best_rounds]["test-r2-mean"]
print("------train-----------------------------------------")
bst = xgb.train(params, dtrain, best_rounds)
print("------predict---------------------------------------")
dtest = xgb.DMatrix(data = x_test)
preds = bst.predict(dtest)
output = pd.DataFrame({'id': test_id.astype(np.int32), 'y': preds})
print("------file generate---------------------------------")
output.to_csv('../upload/cv_' + str(best_rounds) + '_' + str(files_name) + '_' + 'my_preds.csv', index=None)