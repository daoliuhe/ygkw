"""
21/12/2016
Use filtered data to do iterated forecasting

1. training: filtered, testing: filtered, rmse wrt original
    - NN: 0.1760, 0.1836, 0.1818
    - LR: 0.1476
    - SVR: not good
    - Ridge: 0.1485

2. training: filtered, testing: original, rmse wrt original
    - NN: ?
    - LR: ?

3. CONFIRMED: filtfilt does not use future data, but uses padding at both ends.
    - odd: 0.1476
    - even: 0.1476
    - constant: 0.1476
    - filter_size: 5 (0.1476), 10 (0.1301), 15 (0.1416), 20 (0.1520)
            8 (0.1335), 9 (0.1255), 11 (0.1317), this parameter matters.
    - bigger filter size, smoother filtered signal, bigger signal loss

4. NN: 0.1704, where filter_size=9, not as good as LR

5. train_size: 3600 (0.1255), 4000 (0.8585), 4500 (0.1571), 5000 (1.8383)
        5500 (2.2090), 5700 (0.0519), 5730 (0.0454), 5754 (0.0456),
        where filter_size=9, model = lr

6. train_size: 3600 (0.1476), 4000 (0.8599), 4500 (0.1833), 5000 (1.8360)
        5500 (2.2096), 5700 (0.0409), 5730 (0.0397), 5754 (0.0411),
        where filter_size=5, model = lr

22/12/2016
7. normalization not involved.
8. random forest: 0.1723, where filter_size=5, not as good as LR

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
date_start, date_end = '14/Feb/16 00:10:00', '25/Mar/16 00:10:00'
data = slice_data(date_sampled, data_sampled, date_start, date_end)
vibration = data[:, 41]
train_size = 3600
Y_real = vibration[train_size:]  # original vibration signal

vib, _ = filter_data(start=date_start, end=date_end, filter_size=9)  # filtered vibration
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
pl.figure(figsize=(15, 10))
pl.plot(testY[:1000], '-', testY_pred[:1000], '-o')
pl.legend(['Filtered', 'Predicted'], loc='upper left')
pl.show()
