import feat_engineer as bg
import stack_model as sm
import config_params as bp

from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

SEED = 2018
if __name__ == '__main__':


    train, test, y_train, ntrain, ntest, listing_id, listing_id_train = bd.load_base_data() ## I return a lot
    
    y_train = y_train.ravel()
    train_test = pd.concat()
    x_train = " "
    x_test = " "
    
    print(x_train.shape)

#---------------------------------------get_oof_tree(clf, x_train, y_train, x_test, ntrain, ntest, NFOLDS = 5)-------------------------------------------------------
    rf = RandomForestClassifier(n_estimators = 1300, max_depth = 10, criterion='entropy',  n_jobs = -1, min_samples_leaf = 2, random_state=SEED)
    rf_oof_train, rf_oof_test = sm.get_oof_tree(rf, x_train,y_train,x_test,ntrain, ntest)

    gb = GradientBoostingClassifier(learning_rate=0.04, n_estimators = 80, subsample = 0.7, max_depth = 6, min_samples_leaf = 5,random_state=SEED)
    gb_oof_train,gb_oof_test = sm.get_oof_tree(gb,x_train,y_train,x_test,ntrain,ntest)

    et = ExtraTreesClassifier(n_estimators = 250, max_depth= 8, max_features='sqrt',n_jobs = -1)
    et_oof_train, et_oof_test = sm.get_oof_tree(et,x_train,y_train,x_test,ntrain, ntest)

    lr = LogisticRegression(C = 0.02, solver = 'newton-cg', multi_class = 'multinomial', max_iter = 500, n_jobs = -1)
    stda = StandardScaler(with_mean=False)
    processTest = pd.DataFrame(stda.fit_transform(x_test))
    processTrain = pd.DataFrame(stda.transform(x_train))
    lr_oof_train,lr_oof_test = sm.get_oof_tree(lr,processTrain,y_train,processTest,ntrain, ntest)
    # lr_oof_train,lr_oof_test = sm.get_oof_tree(lr,x_train,y_train,x_test,ntrain, ntest)


    # get_oof_regressor(clf, x_train, y_train, x_test, ntrain, ntest, NFOLDS = 5)
    rf = RandomForestRegressor(n_estimators=600, max_depth = 8, n_jobs = -1,  random_state=SEED)
    ada = AdaBoostRegressor(n_estimators=60, learning_rate = 0.01, loss = 'square', random_state=SEED)
    gb = GradientBoostingRegressor(learning_rate=0.02, n_estimators=80, subsample=0.75, max_depth = 6, random_state=SEED)
    et = ExtraTreesRegressor(n_estimators=150, max_depth= 8, max_features='sqrt',n_jobs = - 1, random_state = SEED)
    rf_reg_train, rf_reg_test = sm.get_oof_regressor(rf, x_train, y_train, x_test, ntrain, ntest)
    ada_reg_train, ada_reg_test = sm.get_oof_regressor(ada, x_train, y_train, x_test, ntrain, ntest)
    gb_reg_train, gb_reg_test = sm.get_oof_regressor(gb, x_train, y_train, x_test, ntrain, ntest)
    et_reg_train, et_reg_test = sm.get_oof_regressor(et, x_train, y_train, x_test, ntrain, ntest)
    xgb_params = bp.get_basic(2)
    xgb_reg_train, xgb_reg_test = sm.get_oof_xgb_linear(xgb_params, x_train,y_train,x_test,ntrain,ntest, NFOLDS = 5)

    xgb_params = bp.get_basic(1)
    xgb_oof_train,xgb_oof_test = sm.get_oof_xgb(xgb_params, x_train,y_train,x_test,ntrain,ntest)

    names_one = ['rf','lr','et','xgb']
    names_two = ['high','medium','low']
    train_names = [i + "_" + j for i in names_one for j in names_two]

    x_train = pd.DataFrame(np.hstack([rf_oof_train, lr_oof_train, et_oof_train,xgb_oof_train]))
    x_test =  pd.DataFrame(np.hstack([rf_oof_test, lr_oof_test, et_oof_test,xgb_oof_test]))
    x_train.columns = train_names
    x_test.columns = train_names

    x_train.to_csv("tmp_file/cang_train"+str(SEED)+ "_level_2.csv", index = None)
    x_test.to_csv("tmp_file/cang_test"+str(SEED) + "_level_2.csv", index = None)
