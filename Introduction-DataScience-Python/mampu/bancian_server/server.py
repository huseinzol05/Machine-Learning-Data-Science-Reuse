import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import os
import json
from flask import Flask, render_template, request
from flask_cors import CORS

files = os.listdir(os.getcwd())
print(files)
population2011 = pd.read_csv(files[0])
population2012 = pd.read_csv(files[3])
population2013 = pd.read_csv(files[4])
population2014 = pd.read_csv(files[1])

total_population = pd.concat([population2011, population2012, population2013, population2014])
total_population['Year'] = LabelEncoder().fit_transform(total_population['Year'].astype(str))
total_population['State'] = LabelEncoder().fit_transform(total_population['State'])
total_population['Age Group'] = LabelEncoder().fit_transform(total_population['Age Group'])
total_population['Male (\'000)'] = np.log1p(total_population['Male (\'000)'])
total_population['Female (\'000)'] = np.log1p(total_population['Female (\'000)'])
years = np.unique(total_population['Year'])

params = {}
params['learning_rate'] = 0.1
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'rmse'

x_train, x_test, y_train, y_test = train_test_split(total_population.iloc[:, :3].values, total_population.iloc[:, 4:].values, test_size = 0.2)
d_train_man = lgb.Dataset(x_train, label = y_train[:, 0])
d_valid_man = lgb.Dataset(x_test, label = y_test[:, 0])
watchlist_man = [d_valid_man]
clf_man = lgb.train(params, d_train_man, 1000, watchlist_man)
d_train_woman = lgb.Dataset(x_train, label = y_train[:, 1])
d_valid_woman = lgb.Dataset(x_test, label = y_test[:, 1])
watchlist_woman = [d_valid_woman]
clf_woman = lgb.train(params, d_train_woman, 1000, watchlist_woman)

app = Flask(__name__)
CORS(app)

@app.route('/mampu', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		return ''
	else:
		year = int(request.args.get('year'))
		state = int(request.args.get('state'))
		age = int(request.args.get('age'))
		if int(request.args.get('gender')) == 1:
			array = []
			for i in range(years.shape[0]):
				array.append(np.expm1(total_population.loc[(total_population['Year'] == years[i]) & (total_population['State'] == state) & (total_population['Age Group'] == age)].iloc[0, 4]))
			array.append(np.expm1(clf_man.predict([[year, state, age]])[0]))
			return json.dumps(array)
		else:
			array = []
			for i in range(years.shape[0]):
				array.append(np.expm1(total_population.loc[(total_population['Year'] == years[i]) & (total_population['State'] == state) & (total_population['Age Group'] == age)].iloc[0, 5]))
			array.append(np.expm1(clf_man.predict([[year, state, age]])[0]))
			return json.dumps(array)
		
if __name__ == '__main__':
    app.run(host = '0.0.0.0', threaded = True,  port = 8011)