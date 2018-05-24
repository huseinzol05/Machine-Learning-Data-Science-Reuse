import feat_engineer as feg
import stack_model as sm
import config_params as cp


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

## Regression Model We will Use
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

SEED = 2018

x_train, x_test, y_train, ntrain, ntest = feg.load_base_data():

y_train = y_train.ravel()

print(x_train.shape)

#---------------------------------------get_oof_tree(clf, x_train, y_train, x_test, ntrain, ntest, NFOLDS = 5)------------------------------------------------------

rf = RandomForestRegressor(n_estimators = 500, max_depth = 8, n_jobs = -1,  random_state=SEED)
rf_reg_train, rf_reg_test = sm.get_oof_regressor(rf, x_train, y_train, x_test, ntrain, ntest)

ada = AdaBoostRegressor(n_estimators = 80, learning_rate = 0.004, loss = 'square', random_state=SEED)
ada_reg_train, ada_reg_test = sm.get_oof_regressor(ada, x_train, y_train, x_test, ntrain, ntest)

gb = GradientBoostingRegressor(learning_rate = 0.01, n_estimators=120, subsample = 0.921, max_depth = 5, random_state=SEED)
gb_reg_train, gb_reg_test = sm.get_oof_regressor(gb, x_train, y_train, x_test, ntrain, ntest)

et = ExtraTreesRegressor(n_estimators = 150, max_depth= 6,max_features='sqrt',n_jobs = - 1, random_state = SEED)
et_reg_train, et_reg_test = sm.get_oof_regressor(et, x_train, y_train, x_test, ntrain, ntest)

le = LinearRegression(n_jobs = -1)
le_reg_train, le_reg_test = sm.get_oof_regressor(le, x_train, y_train, x_test, ntrain, ntest)

el = ElasticNet(random_state=SEED)
el_reg_train, el_reg_test = sm.get_oof_regressor(el, x_train, y_train, x_test, ntrain, ntest)

xgb_params = cp.get_config_params(1)
xgb_reg_train, xgb_reg_test = sm.get_oof_xgb_linear(xgb_params, x_train,y_train,x_test,ntrain,ntest, NFOLDS = 5)

lgb_params = cp.get_config_params(2)
lgb_reg_train, lgb_reg_test = sm.get_oof_lgb(lgb_params, x_train,y_train,x_test,ntrain,ntest, NFOLDS = 5)

names_one = ['rf','ada','gb','et','le','el','xgb','lgb']

x_train = pd.DataFrame(np.hstack([rf_reg_train, ada_reg_train, gb_reg_train, et_reg_train, le_reg_train, el_reg_train,
                        xgb_reg_train, lgb_reg_train]))
x_test =  pd.DataFrame(np.hstack([rf_reg_test, ada_reg_test, gb_reg_test, et_reg_test, le_reg_test, el_reg_test, xgb_reg_test, lgb_reg_test]))
x_train.columns = names_one
x_test.columns = names_one

x_train.to_csv("stack_data/train"+str(SEED) + ".csv", index = None)
x_test.to_csv("stack_data/test"+str(SEED) + ".csv", index = None)
