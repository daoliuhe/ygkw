"""
3/1/2017
Test different filters, considering distortion
                rmse
filtfilt        0.1143
savgol          0.1477

4/1/2017: savgol cannot produce better result than filtfilt

"""

from User.functions import rmse, read_data
import pylab as pl
import numpy as np
from User.functions import prepare_data, iterative_forecast, filter_data_, \
    print_progress, direct_forecast
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from scipy.signal import savgol_filter

# date_start, date_end = '14/Aug/16 00:10:00', '25/Sep/16 00:10:00'
data = read_data(year='2016')
vibration = data[:, 41]
train_size = 3600
test_size = len(vibration) - train_size
y_true = vibration[train_size:]  # original vibration signal

horizon = 6
n_y_features = 10  # no. of past vib values
n_hours = int(test_size / horizon)  # for hourly update
y_pred = np.zeros(test_size)
filter_size = 9
tic = time.time()

for hour in range(n_hours):
    start = hour * horizon + 0
    end = hour * horizon + train_size

    vib = vibration[start:end+horizon]
    # vib_filtered = filter_data_(vib, filter_size)
    vib_filtered = savgol_filter(vib, 13, 3, mode='constant')
    trainX, trainY, testX, testY = prepare_data(vib_filtered, train_size)

    model = LinearRegression()
    model.fit(trainX, trainY)  # update model hourly

    # iterative forecasting
    # batch_forecast = iterative_forecast(model, testX, horizon, n_y_features)

    # direct forecasting
    batch_forecast = direct_forecast(model, trainX, trainY, testX[0, :], horizon)

    y_pred[start:start + horizon] = batch_forecast
    print_progress(hour, n_hours, 25)

toc = time.time()
print("\nRunning time: %.4f" % (toc - tic))
test_rmse = rmse(y_true, y_pred)
print('Testing RMSE between y_true (original) and y_pred: %.4f' % test_rmse)
# pl.figure(figsize=(15, 10))
# pl.plot(y_true[:1000], '-', y_pred[:1000], '-o')
# pl.legend(['Original', 'Predicted'], loc='upper left')
# pl.show()