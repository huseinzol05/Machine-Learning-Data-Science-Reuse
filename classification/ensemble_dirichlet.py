import pandas as pd
import numpy as np, numpy.random

df_1 = pd.read_csv('subm_2017-10-26-18-03_lgb_dart_01.csv')
df_2 = pd.read_csv('subm_2017-10-26-18-03_lgb_gbdt_01.csv')
df_3 = pd.read_csv('subm_2017-10-26-18-03_xgb_01.csv')
df_4 = pd.read_csv('subm_2017-10-26-18-04_lgb_dart_02.csv')
df_5 = pd.read_csv('subm_2017-10-26-18-04_lgb_gbdt_02.csv')
df_6 = pd.read_csv('subm_2017-10-26-18-04_xgb_02.csv')
df_7 = pd.read_csv('subm_2017-10-26-18-05_lgb_dart_03.csv')
df_8 = pd.read_csv('subm_2017-10-26-18-05_lgb_gbdt_03.csv')
df_9 = pd.read_csv('subm_2017-10-26-18-05_xgb_03.csv')
df_10 = pd.read_csv('subm_blend009_2017-10-26-18-05.csv')

x = np.random.dirichlet(np.ones(1),size=1)[0]
print x.shape

df_1['is_iceberg'] = x[0] * df_1['is_iceberg'] + x[1] * df_2['is_iceberg'] + x[2] * df_3['is_iceberg'] + x[3] * df_4['is_iceberg'] + x[4] * df_5['is_iceberg'] + x[5] * df_6['is_iceberg'] + x[6] * df_7['is_iceberg'] + x[7] * df_8['is_iceberg'] + x[8] * df_9['is_iceberg'] + x[9] * df_10['is_iceberg']

df_1.to_csv('rank_avg.csv.gz', index = False, compression = 'gzip') 