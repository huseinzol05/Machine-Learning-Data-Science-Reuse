import numpy as np
from sklearn.ensemble import *
import xgboost as xgb
from sklearn.cross_validation import train_test_split

X = np.random.uniform(size=(100,10))
Y = np.random.uniform(size=(100))

# split our dataset for validation
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.2)

# our base models
ada = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)
bagging = BaggingRegressor(n_estimators=500)
et = ExtraTreesRegressor(n_estimators=500)
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, loss='lad',criterion='mse')
rf = RandomForestRegressor(n_estimators=500)

# fit our base models
ada.fit(train_X, train_Y)
bagging.fit(train_X, train_Y)
et.fit(train_X, train_Y)
gb.fit(train_X, train_Y)
rf.fit(train_X, train_Y)

# predict our train set
ada_out_train = ada.predict(train_X)
bagging_out_train = bagging.predict(train_X)
et_out_train = et.predict(train_X)
gb_out_train = gb.predict(train_X)
rf_out_train = rf.predict(train_X)

# predict our test set
ada_out_test = ada.predict(test_X)
bagging_out_test = bagging.predict(test_X)
et_out_test = et.predict(test_X)
gb_out_test = gb.predict(test_X)
rf_out_test = rf.predict(test_X)

# concat column-wise for train
stack_predict_train = np.vstack([ada_out_train,bagging_out_train,et_out_train,gb_out_train,rf_out_train]).T

# concat column-wise for test
stack_predict_test = np.vstack([ada_out_test,bagging_out_test,et_out_test,gb_out_test,rf_out_test]).T

params_xgd = {
  'max_depth': 7,
  'objective': 'reg:linear',
  'learning_rate': 0.033,
  'n_estimators': 1000
}
clf = xgb.XGBRegressor(**params_xgd)
clf.fit(stack_predict_train, train_Y, eval_set=[(stack_predict_test, test_Y)], 
        eval_metric='rmse', early_stopping_rounds=20, verbose=True)

# print mean square error
print(np.mean(np.square(test_Y - clf.predict(stack_predict_test)))
