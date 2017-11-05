"""
23/12/2016
Forward feature selection
Candidate set: y_t-1,..., y_t-10, X_t-1

"""

import pickle
from User.functions import slice_data, rmse
import pylab as pl
import numpy as np
from Dec.func import filter_data, prepare_data_
import time
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import pandas as pd

# ***** use original data *****
data_sampled = pickle.load(open("../Data/sampled2016.p", "rb"))
date_sampled = pickle.load(open("../Data/date2016.p", "rb"))
date_start, date_end = '14/Feb/16 00:10:00', '25/Mar/16 00:10:00'
data = slice_data(date_sampled, data_sampled, date_start, date_end)
y = data[:, 41]

ranked_cols = pickle.load(open("ranked_cols.p", "rb"))
X = data[:, ranked_cols]

train_size = 5000
trainX, trainY, testX, testY = prepare_data_(y, X, train_size)

lr = LinearRegression()
sfs = SFS(lr,
          k_features=(10, 20),
          forward=False,
          floating=False,
          scoring='neg_mean_squared_error',
          cv=5)

sfs = sfs.fit(trainX, trainY)
result = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
result.to_csv('result.csv', encoding='utf-8')
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
pl.grid()
pl.show()
k_index = sfs.k_feature_idx_  # tuple
print('\n')
print(list(k_index))
