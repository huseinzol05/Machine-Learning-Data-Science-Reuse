import lightgbm as lgb
from sk.model_selection import train_test_split

# load any dataset
# preprocessing
# change into vectorizer

# simple k-fold
fold = 5
for i in range(fold):
	# the best parameters i have used all this time
    params = {
        'learning_rate': 0.001,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'eval_metric': 'multi_logloss',
        'sub_feature': 0.5,
        'num_leaves': 100,
        'min_data', 100,
        'min_hessian': 1,
        'num_class': 9,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size = 0.18, random_state = i)
    d_train = lgb.Dataset(x1, y1)
    d_valid = lgb.Dataset(x2, y2)
    watchlist = [d_valid]
	# 3000 epoch, print every 50 epoch, early stop for each 100 epoch
    model = lgb.train(params, d_train, 3000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    if i != 0:
        pred = model.predict(test)
        preds += pred
    else:
        pred = model.predict(test)
        preds = pred.copy()

preds /= fold