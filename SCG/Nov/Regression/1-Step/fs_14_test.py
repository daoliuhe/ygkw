"""
4/11/2016:      best k: 7

"""

import pickle
import numpy as np
from sklearn import preprocessing
import pylab as pl
from sklearn.linear_model import BayesianRidge, LinearRegression
from User.functions import slice_data

# data in a range
data_sampled = pickle.load(open("../Data/sampled2014.p", "rb"))
date_sampled = pickle.load(open("../Data/date2014.p", "rb"))
date_start = '05/Mar/14 00:10:00'
date_end = '14/Apr/14 00:10:00'

data_sliced = slice_data(date_sampled, data_sampled, date_start, date_end)

# selected columns
col_selected = pickle.load(open("selected_cols.p", "rb"))

# --------------------------------------------------
k_best = 7
selected_features = col_selected[:k_best]

outputs = data_sliced[:, 41]
inputs = data_sliced[:, selected_features]

train_size = 3600
slide_size = 144  # data points in 1 day
test_days = 15

Y_real = outputs[3600:5760]
Y_pred = np.zeros((test_days, slide_size))
for day in range(test_days):
    train_from = 1500 + day * slide_size
    train_end = train_size + day * slide_size
    X, Y = inputs[train_from-1:train_end-1, :], outputs[train_from:train_end]

    test_from = 3600 + day * slide_size
    test_end = test_from + slide_size
    tX, tY = inputs[test_from-1:test_end-1, :], outputs[test_from:test_end]

    # scaling
    x_scaler = preprocessing.MinMaxScaler().fit(X)
    trainX = x_scaler.transform(X)
    testX = x_scaler.transform(tX)

    y_scaler = preprocessing.MinMaxScaler().fit(Y.reshape(-1, 1))  # Y is one column, so need reshape
    trainY = y_scaler.transform(Y.reshape(-1, 1))
    testY = y_scaler.transform(tY.reshape(-1, 1))

    # regression
    model = LinearRegression().fit(trainX, trainY)
    testY_slide = model.predict(testX)
    temp = y_scaler.inverse_transform(testY_slide.reshape(-1, 1))
    Y_pred[day, :] = np.ravel(temp)

Y_est = np.ravel(Y_pred.reshape(1, -1))  # convert matrix to vector
test_rmse = np.sqrt(np.mean((Y_real - Y_est) ** 2))
print('The testing RMSE is: %.4f\n' % test_rmse)


# result display
pl.plot(Y_real, '-', Y_est, '-o', linewidth=2.0)
pl.show()

