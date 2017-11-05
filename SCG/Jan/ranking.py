"""
12/1/2017
Feature ranking, copy from Nov/Regression
    a. original data is used
    b. 8 scoring methods are used

14/2 to 25/3            rmse
input: only y           0.1077
input: y + ranking      0.1079 (2), 0.1078 (5)

14/4 to 25/5            rmse
input: only y           0.0806
input: y + ranking      0.0807 (2), 0.0915 (5)

14/6 to 25/7            rmse
input: only y           0.1053
input: y + ranking      0.1055 (2), 0.1065 (5)

Remark:
    a. y inputs only perform best in all cases.
    b. X inputs do not improve accuracy in all cases. UPSET!!!
---------------------------------------------------------------------

Use filtered data for ranking, also worse than y inputs only
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
from User.functions import read_data, col_index, feature_ranking, filter_data_, \
    prepare_data_, print_progress, rmse, direct_forecast
import pickle

FLAG = 1

if FLAG:
    warnings.filterwarnings("ignore")
    # rank features
    start = '14/Feb/16 00:10:00'
    end = '25/Mar/16 00:10:00'
    data = read_data(year='2016', start=start, end=end)
    vib = data[:, 41]
    cols = col_index()
    inputs = data[:, cols]
    ranking_cols = feature_ranking(vib, inputs, cols)
    pickle.dump(ranking_cols, open('ranking_cols.p', 'wb'))
else:
    ranking_cols = pickle.load(open("ranking_cols.p", "rb"))

# data I/O and settings
start = '14/Feb/16 00:10:00'
end = '25/Mar/16 00:10:00'
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

# forecasting
externals = data[:, ranking_cols[:2]]
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
