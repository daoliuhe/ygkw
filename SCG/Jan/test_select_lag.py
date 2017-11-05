"""
6/1/2017
Select optimal number of lagged vibration inputs

"""

from User.functions import rmse, read_data, mape, prepare_data, \
    filter_data_, print_progress, direct_forecast, select_lag
import pylab as pl
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge

data = read_data(year='2016')
vibration = data[:, 41]
train_size = 3600
test_size = len(vibration) - train_size
y_true = vibration[train_size:]  # original vibration signal

horizon = 6
n_y_features = 10  # no. of past vib values
n_hours = int(test_size / horizon)  # for hourly update
y_pred = np.zeros(test_size)
filter_size = 6
tic = time.time()

for hour in range(n_hours):
    start = hour * horizon + 0
    end = hour * horizon + train_size
    vib = vibration[start:end+horizon]
    vib_filtered = filter_data_(vib, filter_size)

    # select best lag every 24 hours
    if hour % 24 == 0:
        n_y_features = select_lag(vib_filtered, train_size, horizon)
        print(n_y_features)

    trainX, trainY, testX, testY = prepare_data(vib_filtered, train_size, n_y_features)
    model = LinearRegression()
    model.fit(trainX, trainY)
    # direct forecasting
    batch_forecast = direct_forecast(model, trainX, trainY, testX[0, :], horizon)

    y_pred[start:start + horizon] = batch_forecast
    print_progress(hour, n_hours, 25)

toc = time.time()
print("\nRunning time: %.4f" % (toc - tic))
test_rmse = rmse(y_true, y_pred)
test_mape = mape(y_true, y_pred)
print('RMSE between y_true (original) and y_pred: %.4f' % test_rmse)
# print('MAPE between y_true (original) and y_pred: %.4f' % test_mape)
# pl.figure(figsize=(15, 10))
# pl.plot(y_true[:400], '-', y_pred[:400], '-o')
# pl.legend(['Original', 'Predicted'], loc='upper left')
# pl.show()
