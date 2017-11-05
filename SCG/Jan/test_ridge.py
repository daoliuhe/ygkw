"""
17/1/2017
Ridge imposes constraints on weights, so it can tackle collinearity among inputs.
It is supposed to be better than linear regression.

Settings: horizon=6, n_y_input=20, filter_size=6
    linear: 0.1046
    ridge: 0.1271 (1), 0.1047 (.00004), 0.1046 (.000000004)
    ridgeCV: 0.1138

Remark:
    a. linear is better than ridge
    b. ridge with smaller alpha tends to produce better accuracy. Note that ridge
    becomes linear if alpha is very small. The penalty is gone.
"""

from User.functions import rmse, read_data, mape, prepare_data, \
    filter_data_, print_progress, direct_forecast
import numpy as np
import time
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
start = '14/Feb/16 00:10:00'
end = '25/Mar/16 00:10:00'
data = read_data(year='2016', start=start, end=end)
vibration = data[:, 41]
train_size = 3600
test_size = len(vibration) - train_size
y_true = vibration[train_size:]  # original vibration signal

horizon = 6
n_y_features = 20  # no. of past vib values
n_hours = int(test_size / horizon)  # for hourly update
y_pred = np.zeros(test_size)
filter_size = 6
tic = time.time()
alpha_list = [2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4]

for hour in range(n_hours):
    start = hour * horizon + 0
    end = hour * horizon + train_size
    vib = vibration[start:end+horizon]
    vib_filtered = filter_data_(vib, filter_size)

    trainX, trainY, testX, testY = prepare_data(vib_filtered, train_size, n_y_features)
    model = Ridge(alpha=16)
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
print('MAPE between y_true (original) and y_pred: %.4f' % test_mape)