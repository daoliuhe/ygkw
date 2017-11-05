"""
29/12/2016
Check border distortion's effect on prediction
Based on ms_denoise.py

In reality, new data comes from time to time. Hence, filtering is performed hourly,
i.e. after receiving every 6 new measurements. Data filtering at the end will have distortion.
As tested, for the new data points, 4 of 6 distorts (depends on filter size).

3/1/2017
Data is filtered every hour by incorporating new measurements. Distortion occurs.
Case I: distortion not considered, ms_denoise.py
Case II: distortion considered, ms_distortion.py
Remark:
a. distortion changes the result of iterative forecasting
b. distortion may change the result of direct forecasting, it depends on the filter size
c. filter size changes the no. of distorted data, and therefore change direct forecasting result
d. overall, forecasting is still good with distortion

Test on different periods of data, with distortion, iterated forecasting
start           end             without     with (fs=9) with (fs=5)     direct5     direct9
14/Feb/16       25/Mar/16       0.1255      0.1245      0.1457          0.1198      0.1143
14/Apr/16       25/May/16       0.1109      0.1144      0.1328          0.0909      0.0854
14/Jun/16       25/Jul/16       0.1475      0.1515      0.2081          0.1162      0.1098
14/Aug/16       25/Sep/16       0.2525      0.2838      0.3122          0.1355      0.1300
Remark: direct is better than iterated, 9 is better than 5


"""

from User.functions import rmse, read_data, mape
import pylab as pl
import numpy as np
from User.functions import prepare_data, iterative_forecast, filter_data_, \
    print_progress, direct_forecast
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

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
    vib_filtered = filter_data_(vib, filter_size)
    trainX, trainY, testX, testY = prepare_data(vib_filtered, train_size)

    model = LinearRegression()
    # model = MLPRegressor()
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
test_mape = mape(y_true, y_pred)
print('Testing RMSE between y_true (original) and y_pred: %.4f' % test_rmse)
pl.figure(figsize=(15, 10))
pl.plot(y_true[:1000], '-', y_pred[:1000], '-o')
pl.legend(['Original', 'Predicted'], loc='upper left')
# pl.show()
