import numpy as np
import pandas as pd


xgb_params_level1 = {
    'objective': 'reg:linear',
    'metric': 'rmse',
    'max_depth' : 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'seed': 0,
    'nthread': -1,
    'verbose':0
    }


lgb_params_regressor = {
	"booster": 'gbdt',
	'metric': 'rmse',
    'objective': 'regression',
	"subsample":0.8,
    'num_leaves': 2 ** 5,
	"colsample_bytree":0.7,
	"colsample_bylevel":0.7,
	'learning_rate': 0.01,
    'reg_lambda': 0.05,
    'verbose': 0,
    'nthread':-1
}

xgb_params_level2 = {
        'eta':0.01,
        'colsample_bytree':.8,
        'subsample':.75,
        'nthread':16,
        'max_depth':4,
        'objective':'multi:softprob',
        'eval_metric':'mlogloss',
        'num_class':3,
        'silent':1
        }

def get_config_params(level_step):
	if level_step == 1:
		return xgb_params_level1
	if level_step == 2:
		return lgb_params_regressor
	if level_step == 3:
		return xgb_params_level2