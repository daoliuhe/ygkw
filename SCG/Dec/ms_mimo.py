"""
19/12/2016
MIMO

21/12/2016
I think MIMO and direct are the same due to the processing pattern of sklearn
"""

import pickle
from User.functions import slice_data, rmse
import pylab as pl
from sklearn.svm import SVR
import numpy as np
from Dec.func import prepare_data, mimo_data, print_progress
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

data_sampled = pickle.load(open("../Data/sampled2016.p", "rb"))
date_sampled = pickle.load(open("../Data/date2016.p", "rb"))
date_start, date_end = '14/Feb/16 00:10:00', '25/Mar/16 00:10:00'
data = slice_data(date_sampled, data_sampled, date_start, date_end)
vibration = data[:, 41]

"""
# ***** without hourly update *****
train_size = 3600
train_series = vibration[:train_size]
n_features = 10
horizon = 6
trainX, trainY = mimo_data(train_series, n_features, horizon)

# model = LinearRegression()
# model = MLPRegressor()
# model = RandomForestRegressor()

model.fit(trainX, trainY)

test_series = vibration[train_size-n_features:]
testX, testY = mimo_data(test_series, n_features, horizon)
ind = range(0, len(testY), horizon)
testX_batch = testX[ind, :]
y_pred = model.predict(testX_batch).reshape(1, -1)
Y_pred = np.ravel(y_pred)

a, b, c, Y_real = prepare_data(vibration, train_size)
test_rmse = rmse(Y_real, Y_pred)
print('Testing RMSE: %.4f' % test_rmse)
pl.plot(Y_real, '-', Y_pred, '-o')
pl.show()
"""

# model hourly update
train_size = 3600
a, b, c, Y_real = prepare_data(vibration, train_size)
n_features = 10
horizon = 6
n_hours = int(len(Y_real) / horizon)
Y_pred = np.zeros(len(Y_real))
tic = time.time()
for hour in range(n_hours):
    start = hour * horizon + 0
    train_series = vibration[start:(train_size + start)]
    trainX, trainY = mimo_data(train_series, n_features, horizon)

    model = LinearRegression()
    # model = MLPRegressor()
    # model = RandomForestRegressor()

    model.fit(trainX, trainY)

    test_series = vibration[(start + train_size - n_features):]
    testX, testY = mimo_data(test_series, n_features, horizon)
    testX_batch = testX[0, :].reshape(1, -1)
    testY_pred = model.predict(testX_batch)
    Y_pred[start:start+horizon] = testY_pred
    print_progress(hour, n_hours, 25)

toc = time.time()

print("Running time: %.4f" % (toc - tic))
test_rmse = rmse(Y_real, Y_pred)
print('\nTesting RMSE: %.4f' % test_rmse)
pl.plot(Y_real, '-', Y_pred, '-o')
pl.show()