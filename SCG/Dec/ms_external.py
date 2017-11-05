"""
22/12/2016
1. Inputs: vibration (10 lags) + external variables
2. No feature selection, just pick top ones directly

3. Adding external variables improves accuracy.
4. 1-step accuracy is very important, which can lead to good multi step accuracy
5. NN very bad
6. to make a ranking list, horizon not affect accuracy
7. X lag matters a lot, lag 1 deliver good 1-step accuracy, see 4
8. the inclusion of historical vibration data make external variables less important
9. try forward feature selection

"""

import pickle
from User.functions import slice_data, rmse
import pylab as pl
import numpy as np
from Dec.func import prepare_data, iterative_forecast, filter_data, \
    print_progress, prepare_data_, iterative
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# -------------------------- function definitions end here --------------------------

data_sampled = pickle.load(open("../Data/sampled2016.p", "rb"))
date_sampled = pickle.load(open("../Data/date2016.p", "rb"))
date_start, date_end = '14/Feb/16 00:10:00', '25/Mar/16 00:10:00'
data = slice_data(date_sampled, data_sampled, date_start, date_end)
vibration = data[:, 41]
train_size = 3600
Y_real = vibration[train_size:]  # original vibration signal

k_best = 5
col_ind = pickle.load(open("../Nov/Regression/col_index_new.p", "rb"))
ranked_cols = pickle.load(open("ranked_cols.p", "rb"))
selected_features = np.array(col_ind)[ranked_cols][:k_best]
# selected_features = ranked_cols[:k_best]
inputs = data[:, selected_features]  # external variables

vib, _ = filter_data(filter_size=9, start=date_start, end=date_end)  # filtered vibration
trainX, trainY, testX, testY = prepare_data_(vib, inputs, train_size)

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
    batch_forecast = iterative(model, testX_batch, horizon)
    testY_pred[start:start+horizon] = batch_forecast
    print_progress(hour, n_hours, 25)

toc = time.time()
print("\nRunning time: %.4f" % (toc - tic))
test_rmse1 = rmse(testY, testY_pred)
test_rmse2 = rmse(Y_real, testY_pred)
print('Testing RMSE between testY (filtered) and y_pred: %.4f' % test_rmse1)
print('Testing RMSE between Y_real (original) and y_pred: %.4f' % test_rmse2)
pl.figure(figsize=(15, 10))
pl.plot(Y_real[:1000], '-', testY_pred[:1000], '-o')
pl.legend(['Original', 'Predicted'], loc='upper left')
# pl.figure(figsize=(15, 10))
# pl.plot(testY[:1000], '-', y_pred[:1000], '-o')
# pl.legend(['Filtered', 'Predicted'], loc='upper left')
pl.show()
