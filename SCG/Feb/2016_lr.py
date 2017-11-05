""" 15/02/2017 copy from Jan """

from User.functions import rmse, read_data, mape, prepare_data, \
    filter_data_, print_progress, direct_forecast, iterative_forecast
import pylab as pl
import numpy as np
import time
from sklearn.linear_model import LinearRegression

start = '14/Feb/16 00:10:00'
end = '25/Mar/16 00:10:00'
data = read_data(year='2016', start=start, end=end)
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

    trainX, trainY, testX, testY = prepare_data(vib_filtered, train_size, n_y_features)
    model = LinearRegression()

    # iterative forecasting
    model.fit(trainX, trainY)
    batch_forecast = iterative_forecast(model, testX, horizon, n_y_features)

    # direct forecasting
    # batch_forecast = direct_forecast(model, trainX, trainY, testX[0, :], horizon)

    y_pred[start:start + horizon] = batch_forecast
    print_progress(hour, n_hours, 25)

toc = time.time()
print("\nRunning time: %.4f" % (toc - tic))
test_rmse = rmse(y_true, y_pred)
test_mape = mape(y_true, y_pred)
print('RMSE between y_true (original) and y_pred: %.4f' % test_rmse)
print('MAPE between y_true (original) and y_pred: %.4f' % test_mape)
pl.figure(figsize=(15, 10))
pl.plot(y_true[:200], '-', y_pred[:200], '-o')
pl.legend(['Original', 'Predicted'], loc='upper left')
pl.show()
