"""
16/12/2016
Try multi-step forecasting:
    a. iterative
    b. direct
    c. iterate-direct hybrid
    d. MIMO

Setting:
    a. horizon = 6
    b. n_features = 10

19/12/2016
    a. RMSE compare:
            1. persist: 0.1987
            2. svr: 0.2494
            3. lr: 0.1978
            4. nn: 0.1948


"""

import pickle
from User.functions import slice_data, rmse
import pylab as pl
from sklearn.svm import SVR
import numpy as np
from Dec.func import prepare_data, iterative_forecast, print_progress
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

data_sampled = pickle.load(open("../Data/sampled2016.p", "rb"))
date_sampled = pickle.load(open("../Data/date2016.p", "rb"))
date_start, date_end = '14/Feb/16 00:10:00', '25/Mar/16 00:10:00'
data = slice_data(date_sampled, data_sampled, date_start, date_end)
vibration = data[:, 41]

train_size = 3600
trainX, trainY, testX, testY = prepare_data(vibration, train_size)

horizon = 6
n_features = 10
n_hours = int(len(testY) / horizon)  # for hourly update
X = np.vstack((trainX, testX))
Y = np.hstack((trainY, testY))
train_len = len(trainY)
testY_pred = np.zeros(len(testY))
tic = time.time()

# ----------------------------------
# with normalization
# for hour in range(n_hours):
#     start = hour * horizon + 0
#     end = hour * horizon + train_len
#     trainX_new = X[start:end, :]
#     trainY_new = Y[start:end]
#
#     x_scaler = MinMaxScaler().fit(trainX_new)
#     trainX_new_scale = x_scaler.transform(trainX_new)
#     y_scaler = MinMaxScaler().fit(trainY_new.reshape(-1, 1))
#     trainY_new_scale = y_scaler.transform(trainY_new.reshape(-1, 1))
#     model = SVR()
#     model.fit(trainX_new_scale, np.ravel(trainY_new_scale))  # update model hourly
#
#     testX_batch = testX[start:start+horizon, :]
#     testX_batch_scale = x_scaler.transform(testX_batch)
#     batch_forecast = iterative_forecast(model, testX_batch_scale, horizon, n_features)
#     temp = y_scaler.inverse_transform(batch_forecast.reshape(-1, 1))
#     y_pred[start:start+horizon] = np.ravel(temp)
# ----------------------------------

# without norlization
for hour in range(n_hours):
    start = hour * horizon + 0
    end = hour * horizon + train_len
    trainX_new = X[start:end, :]
    trainY_new = Y[start:end]
    # model = SVR()
    model = LinearRegression()
    # model = MLPRegressor()
    model.fit(trainX_new, trainY_new)  # update model hourly

    testX_batch = testX[start:start+horizon, :]
    batch_forecast = iterative_forecast(model, testX_batch, horizon, n_features)
    testY_pred[start:start+horizon] = batch_forecast
    print_progress(hour, n_hours, 25)
# ----------------------------------

toc = time.time()
print("Running time: %.4f" % (toc - tic))
test_rmse = rmse(testY, testY_pred)
print('Testing RMSE: %.4f' % test_rmse)
pl.figure(figsize=(15, 10))
pl.plot(testY[:500], '-', testY_pred[:500], '-o')
pl.show()

