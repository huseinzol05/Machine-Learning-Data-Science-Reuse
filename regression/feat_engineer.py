import numpy as np
import pandas as pd

def load_base_data():
	'''
	You can do your Feature engineering in this place
	x_train: x_data without lable
	x_test: you need to predict
	y_train: x_data label
	n_train: train column length
	n_test: test column length
	test_id
	'''
	train = pd.read_csv('../input/train.csv')
	test = pd.read_csv('../input/test.csv')

	#--------------------------------------------------------main model---------------------------------------------------------------
	x_train = train_test[0:ntrain]
	x_test = train_test[ntrain:]
	return x_train, x_test, y_train, ntrain, ntest