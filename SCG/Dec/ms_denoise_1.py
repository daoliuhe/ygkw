"""
22/12/2016
Test on different periods of data

start           end             rmse
14/Feb/16       25/Mar/16       0.1255
14/Apr/16       25/May/16       0.1109
14/Jun/16       25/Jul/16       0.1475
14/Aug/16       25/Sep/16       0.2525

Remark: result looks good.
"""

import pickle
from User.functions import slice_data, rmse
import pylab as pl
from sklearn.svm import SVR
import numpy as np
from Dec.func import prepare_data, iterative_forecast, filter_data, \
    print_progress, direct_forecast
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

data_sampled = pickle.load(open("../Data/sampled2016.p", "rb"))
date_sampled = pickle.load(open("../Data/date2016.p", "rb"))
date_start, date_end = '14/Aug/16 00:10:00', '25/Sep/16 00:10:00'
data = slice_data(date_sampled, data_sampled, date_start, date_end)
vibration = data[:, 41]
train_size = 3600
Y_real = vibration[train_size:]  # original vibration signal

vib, _ = filter_data(filter_size=9, start=date_start, end=date_end)  # filtered vibration
trainX, trainY, testX, testY = prepare_data(vib, train_size)

horizon = 6
n_features = 10
n_hours = int(len(testY) / horizon)  # for hourly update
X = np.vstack((trainX, testX))
Y = np.hstack((trainY, testY))
train_len = len(trainY)
testY_pred = np.zeros(len(testY))
tic = time.time()

for hour in range(n_hours):
    start = hour * horizon + 0
    end = hour * horizon + train_len
    trainX_new = X[start:end, :]
    trainY_new = Y[start:end]
    # model = MLPRegressor(early_stopping=True)
    model = LinearRegression()
    model.fit(trainX_new, trainY_new)  # update model hourly

    testX_batch = testX[start:start+horizon, :]
    batch_forecast = iterative_forecast(model, testX_batch, horizon, n_features)
    testY_pred[start:start+horizon] = batch_forecast
    print_progress(hour, n_hours, 25)
# ----------------------------------

toc = time.time()

print("\nRunning time: %.4f" % (toc - tic))
test_rmse1 = rmse(testY, testY_pred)
test_rmse2 = rmse(Y_real, testY_pred)
print('Testing RMSE between testY (filtered) and y_pred: %.4f' % test_rmse1)
print('Testing RMSE between Y_real (original) and y_pred: %.4f' % test_rmse2)
pl.figure(figsize=(15, 10))
pl.plot(Y_real[:1000], '-', testY_pred[:1000], '-o')
pl.legend(['Original', 'Predicted'], loc='upper left')
pl.title('Original vs Predicted')
pl.figure(figsize=(15, 10))
pl.plot(testY[:1000], '-', testY_pred[:1000], '-o')
pl.legend(['Filtered', 'Predicted'], loc='upper left')
pl.title('Filtered vs Predicted')
pl.show()
