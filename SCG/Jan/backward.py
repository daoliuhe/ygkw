"""
12/1/2017: backward feature selection

14/2 to 25/3            rmse
input: only y           0.1077
input: y + backward     0.1079 (25), data_=data[:3600, :]
input: y + backward     0.1081 (12), data_=data[:5000, :]

14/4 to 25/5            rmse
input: only y           0.0806
input: y + backward     0.0810 (6), data_=data[:3600, :]
input: y + backward     0.0808 (6), data_=data[:5000, :]

14/6 to 25/7            rmse
input: only y           0.1053
input: y + backward     0.1073 (26), data_=data[:3600, :]
input: y + backward     0.1079 (29), data_=data[:5000, :]

Remark:
    a. y inputs only perform best in all cases.
    b. X inputs do not improve accuracy in all cases. UPSET!!!
"""

from User.functions import read_data, rmse, prepare_data_, \
    filter_data_, direct_forecast, forward_selection, print_progress, backward_selection
from sklearn.linear_model import LinearRegression
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
import time

# data I/O and settings
start = '14/Jun/16 00:10:00'
end = '25/Jul/16 00:10:00'
data = read_data(year='2016', start=start, end=end)
vibration = data[:, 41]
train_size = 3600
test_size = len(vibration) - train_size
y_true = vibration[train_size:]  # original vibration signal
y_pred = np.zeros(test_size)
n_y_input = 10
horizon = 6
n_hours = int(test_size / horizon)  # for hourly update
filter_size = 6

# select external variables
data_ = data[:5000, :]
k, selected_cols = backward_selection(data_, n_y_input, plot=True)
print('\nNumber of selected external features: %s' % k)
print('Selected columns are: %s\n' % selected_cols)
# pl.show()

# forecasting
# selected_cols = [2, 5, 7, 12, 13, 15, 16, 19, 23, 29, 30, 33, 35, 39, 44, 46, 47, 50, 52, 54, 55, 59, 61, 62, 63]
externals = data[:, selected_cols]
for hour in range(n_hours):
    start = hour * horizon + 0
    end = hour * horizon + train_size
    vib = vibration[start:end+horizon]
    vib_filtered = filter_data_(vib, filter_size)
    X_input = externals[start:end+horizon, :]

    trainX, trainY, testX, testY = prepare_data_(vib_filtered, X_input,
                                                 train_size, n_y_input)
    model = LinearRegression()
    model.fit(trainX, trainY)
    # direct forecasting
    batch_forecast = direct_forecast(model, trainX, trainY, testX[0, :], horizon)

    y_pred[start:start + horizon] = batch_forecast
    print_progress(hour, n_hours, 25)

test_rmse = rmse(y_true, y_pred)
print('RMSE between y_true (original) and y_pred: %.4f' % test_rmse)