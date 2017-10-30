import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# test files

test_xgb = pd.read_csv('test_sub_xgb.csv')
test_lgb = pd.read_csv('test_sub_lgb.csv')
test_dnn = pd.read_csv('test_dnn_predictions.csv')

test=pd.read_csv('test.csv')


test = pd.concat([test, 
                   test_xgb[['target']].rename(columns = {'target' : 'xgb'}),
                   test_lgb[['target']].rename(columns = {'target' : 'lgb'}),
                   test_dnn[['target']].rename(columns = {'target' : 'dnn'})
                  ], axis = 1)


train_cols = ['xgb', 'lgb', 'dnn']


# In[ ]:


### preprocess


# In[ ]:


for t in train_cols:
    test[t + '_rank'] = test[t].rank()


test['target'] = (test['xgb_rank'] + test['lgb_rank'] + test['dnn_rank']) / (3 * test.shape[0])


# # The final submission

# In[ ]:


test[['id', 'target']].to_csv('rank_avg.csv.gz', index = False, compression = 'gzip') 

