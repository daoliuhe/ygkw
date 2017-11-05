"""
15/12/2016
best k: 25

"""

import pickle
import numpy as np
from sklearn import preprocessing
import pylab as pl
from sklearn.linear_model import BayesianRidge, LinearRegression
from Dec.func import filter_data
from sklearn.svm import SVR

vib, inputs = filter_data()  # filtered data

# --------------------------------------------------
k_best = 25
ranked_cols = pickle.load(open("ranked_cols.p", "rb"))
selected_features = ranked_cols[:k_best]

outputs = vib
inputs = inputs[:, selected_features]
# inputs = np.hstack((inputs, vib.reshape(-1, 1)))
horizon = 6  # forecasting horizon, e.g. 6-step (60 min) ahead

train_size = 3600
slide_size = 144  # data points in 1 day
test_days = 15

Y_real = outputs[train_size:5760]
Y_pred = np.zeros((test_days, slide_size))
for day in range(test_days):
    train_from = 10 + day * slide_size
    train_end = train_size + day * slide_size
    X, Y = inputs[train_from-horizon:train_end-horizon, :], outputs[train_from:train_end]

    test_from = train_size + day * slide_size
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
    model = SVR(C=0.5)
    model.fit(trainX, trainY)
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
pl.plot(Y_real[1870:1940], '-', Y_est[1870:1940], '-o', linewidth=2.0)
pl.legend(['Real spike', 'Predicted spike'])
pl.show()

