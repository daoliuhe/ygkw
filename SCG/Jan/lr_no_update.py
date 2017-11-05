"""
18/1/2017
LinearRegression and MLPRegressor, no hourly update, for compare

Remark:
    a. without hourly update, linear still produce very close result, wonderful!!!
    b. linear: 0.1094, MLP: 0.2247, 0.2110, 0.1348, 0.1480
"""

from User.functions import rmse, read_data, mape, prepare_data, \
    filter_data_, print_progress, direct_nn
import pylab as pl
import numpy as np
import time
import pyrenn as nn
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

start = '14/Feb/16 00:10:00'
end = '25/Mar/16 00:10:00'
data = read_data(year='2016', start=start, end=end)
vibration = data[:, 41]
train_size = 3600
test_size = len(vibration) - train_size
y_true = vibration[train_size:]  # original vibration signal

horizon = 48
n_y_input = 10  # no. of past vib values
filter_size = 6
n_hours = int(test_size / horizon)
y_pred = np.zeros(test_size)

vib_filtered = filter_data_(vibration, filter_size)
trainX, trainY, testX, testY = prepare_data(vib_filtered, train_size, n_y_input)

# direct forecasting models
n = len(trainY)
n_samples = n - horizon + 1
model = {}
for h in range(horizon):
    X = trainX[0:n_samples, :]
    y = trainY[(0 + h):(n_samples + h)]
    # mdl = LinearRegression()
    mdl = MLPRegressor(hidden_layer_sizes=(10,), solver='lbfgs')
    model[h] = mdl.fit(X, y)

# testing
for hour in range(n_hours):
    start = hour * horizon + 0
    forecast = np.zeros(horizon)
    for h in range(horizon):
        forecast[h] = model[h].predict(testX[start, :].reshape(1, -1))
    y_pred[start:start + horizon] = forecast
    print_progress(hour, n_hours, 25)

test_rmse = rmse(y_true, y_pred)
test_mape = mape(y_true, y_pred)
print('RMSE between y_true (original) and y_pred: %.4f' % test_rmse)
print('MAPE between y_true (original) and y_pred: %.4f' % test_mape)
pl.figure(figsize=(15, 10))
pl.plot(y_true[:200], '-', y_pred[:200], '-o')
pl.legend(['Original', 'Predicted'], loc='upper left')
pl.show()
