"""
4/1/2017: Test on 2014 data

1. 05/Mar/14 to 14/Apr/14
persistence: rmse = 0.5399

filter size     iterated        direct
    9           0.5663          0.3920
    5           0.7793          0.4322
    7           0.5431          0.3543
    15                          0.5059
    11          0.6057          0.4506
    13                          0.4834
    6                           0.3424      best
    no filter   0.5438          0.5468
Remark:
    a. from rmse, result is not good. from plot, result is good. prediction is
    quite good in non-spike area, and bad in spike area.
    b. filter removes spike, so prediction near spike is not good.
    c. the bigger filter size, the smoother the filtered data, the bigger the
    prediction error in spike area.
    d. best filter size: 6, filter size can be odd and even numbers

5/1/2017 -------------------------------------------
2. test on different periods
start           end             direct6     direct9
05/Mar/14       14/Apr/14       0.3424      0.3920
05/Apr/14       14/May/14       0.0962      0.1078
05/May/14       14/Jun/14       0.1594      0.1719
Remark: results are very good
"""

from User.functions import rmse, read_data, mape, prepare_data, \
    iterative_forecast, filter_data_, print_progress, direct_forecast
import pylab as pl
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge

date_start, date_end = '05/May/14 00:10:00', '14/Jun/14 00:10:00'
data = read_data(year='2014', start=date_start, end=date_end)
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
    model.fit(trainX, trainY)

    # iterative forecasting
    # batch_forecast = iterative_forecast(model, testX, horizon, n_y_features)

    # direct forecasting
    batch_forecast = direct_forecast(model, trainX, trainY, testX[0, :], horizon)

    y_pred[start:start + horizon] = batch_forecast
    print_progress(hour, n_hours, 25)

toc = time.time()
print("\nRunning time: %.4f" % (toc - tic))
test_rmse = rmse(y_true, y_pred)
test_mape = mape(y_true, y_pred)
print('RMSE between y_true (original) and y_pred: %.4f' % test_rmse)
print('MAPE between y_true (original) and y_pred: %.4f' % test_mape)
pl.figure(figsize=(15, 10))
pl.plot(y_true, '-', y_pred, '-o')
pl.legend(['Original', 'Predicted'], loc='lower left')
pl.show()
