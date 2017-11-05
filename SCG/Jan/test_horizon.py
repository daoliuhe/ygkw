"""
5/1/2017: horizon vs rmse
filter size (6) and direct forecasting are used
Remark: error goes high with increasing horizon
"""

from User.functions import rmse, read_data, prepare_data, \
    filter_data_, print_progress, direct_forecast
import pylab as pl
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

data = read_data(year='2016')
vibration = data[:, 41]
train_size = 3600
test_size = len(vibration) - train_size
y_true = vibration[train_size:]  # original vibration signal

n_y_features = 10  # no. of past vib values
y_pred = np.zeros(test_size)
filter_size = 6

test_rmse = []
for horizon in range(1, 19):
    n_hours = int(test_size / horizon)  # for hourly update
    for hour in range(n_hours):
        start = hour * horizon + 0
        end = hour * horizon + train_size
        vib = vibration[start:end + horizon]
        vib_filtered = filter_data_(vib, filter_size)

        trainX, trainY, testX, testY = prepare_data(vib_filtered, train_size, n_y_features)
        model = LinearRegression()
        model.fit(trainX, trainY)
        # direct forecasting
        batch_forecast = direct_forecast(model, trainX, trainY, testX[0, :], horizon)
        y_pred[start:start + horizon] = batch_forecast
        print_progress(hour, n_hours, 25)

    test_rmse.append(rmse(y_true, y_pred))
    print('horizon=%d: rmse=%.4f' % (horizon, rmse(y_true, y_pred)))

x = list(range(1, 19))
y = test_rmse
s = pd.Series(y, index=x)
print('\n')
print(s)
pl.plot(x, y, linewidth=2.0)
pl.xlabel('Forecasting horizon')
pl.ylabel('Testing RMSE')
pl.show()
