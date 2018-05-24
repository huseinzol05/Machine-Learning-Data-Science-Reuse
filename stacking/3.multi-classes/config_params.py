import numpy as np
import pandas as pd


xgb_params_level1 = {
        'eta':0.01,
        'colsample_bytree':.8,
        'subsample':.8,
        'seed':2017,
        'nthread':16,
        'max_depth':6,
        'objective':'multi:softprob',
        'eval_metric':'mlogloss',
        'num_class':3,
        'silent':1
        }

xgb_params_regressor = {
	"booster": 'gbtree',
	"eval_metric":'rmse',
	"gamma":0.1,
	"min_child_weight":1.5,
	"max-depth":5,
	"lambda":10,
	"subsample":0.7,
	"colsample_bytree":0.7,
	"colsample_bylevel":0.7,
	"eta":0.02,
}

xgb_params_level2 = {
            'eta':0.01,
            'colsample_bytree':.8,
            'subsample':.75,
            'seed':2018,
            'nthread':16,
            'max_depth':4,
            'objective':'multi:softprob',
            'eval_metric':'mlogloss',
            'num_class':3,
            'silent':1
            }

def get_basic(level_step):
	if level_step == 1:
		return xgb_params_level1
	if level_step == 2:
		return xgb_params_regressor
	if level_step == 3:
		return xgb_params_level2
