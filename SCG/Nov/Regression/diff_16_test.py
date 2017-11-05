# -*- coding: utf-8 -*-
"""
@author: 40204849
Created on 10/11/2016

"""

import pickle
import numpy as np
from sklearn import preprocessing
import pylab as pl
from sklearn.linear_model import BayesianRidge, LinearRegression
from User.functions import slice_data
from sklearn.svm import SVR

# data in a range
data_sampled = pickle.load(open("../Data/sampled2016.p", "rb"))
date_sampled = pickle.load(open("../Data/date2016.p", "rb"))
date_start = '14/Feb/16 00:10:00'
date_end = '25/Mar/16 00:10:00'

data_sliced = slice_data(date_sampled, data_sampled, date_start, date_end)

# selected columns
col_selected = pickle.load(open("selected_cols.p", "rb"))

# --------------------------------------------------
k_best = 5
selected_features = col_selected[:k_best]
# add y_t-6
# selected_features = np.concatenate(([41], selected_features), axis=0)

outputs = data_sliced[:, 41]
outputs = outputs - np.hstack((outputs[0], outputs[:-1]))
inputs = data_sliced[:, selected_features]
horizon = 1  # forecasting horizon, e.g. 6-step (60 min) ahead

train_size = 3600
slide_size = 144  # data points in 1 day
test_days = 15

Y_real = outputs[3600:5760]
Y_pred = np.zeros((test_days, slide_size))
for day in range(test_days):
    train_from = 1000 + day * slide_size
    train_end = train_size + day * slide_size
    X, Y = inputs[train_from-horizon:train_end-horizon, :], outputs[train_from:train_end]

    test_from = 3600 + day * slide_size
    test_end = test_from + slide_size
    tX, tY = inputs[test_from-horizon:test_end-horizon, :], outputs[test_from:test_end]

    # scaling
    x_scaler = preprocessing.MinMaxScaler().fit(X)
    trainX = x_scaler.transform(X)
    testX = x_scaler.transform(tX)

    y_scaler = preprocessing.MinMaxScaler().fit(Y.reshape(-1, 1))  # Y is one column, so need reshape
    trainY = y_scaler.transform(Y.reshape(-1, 1))
    testY = y_scaler.transform(tY.reshape(-1, 1))

    # regression
    model = SVR().fit(trainX, trainY)
    testY_slide = model.predict(testX)
    temp = y_scaler.inverse_transform(testY_slide.reshape(-1, 1))
    Y_pred[day, :] = np.ravel(temp)

Y_est = np.ravel(Y_pred.reshape(1, -1))  # convert matrix to vector
test_rmse = np.sqrt(np.mean((Y_real - Y_est) ** 2))
print('The testing RMSE is: %.4f\n' % test_rmse)

# result display
pl.figure()
pl.plot(Y_real, '-', Y_est, '-o', linewidth=2.0)
pl.figure()
pl.plot(Y_real[1870:1940], '-s', Y_est[1870:1940], '-o', linewidth=2.0)

pl.show()

