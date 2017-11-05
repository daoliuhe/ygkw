"""
7/2/2017
Linear Regression: benchmark model
1-step ahead forecasting: 0.1021

"""

from User.functions import rmse, read_data, prepare_data, filter_data_
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pylab as pl

start = '14/Feb/16 00:10:00'
end = '25/Mar/16 00:10:00'
data = read_data(year='2016', start=start, end=end)
vibration = data[:, 41]
train_size = 3600
test_size = len(vibration) - train_size
y_true = vibration[train_size:]  # original vibration signal

horizon = 6
n_y_input = 10  # no. of past vib values
filter_size = 6
n_hours = int(test_size / horizon)

vib_filtered = filter_data_(vibration, filter_size)
trainX, trainY, testX, testY = prepare_data(vib_filtered, train_size, n_y_input)

model = LinearRegression()
model.fit(trainX, trainY)
y_pred = model.predict(testX)

test_rmse = rmse(y_true, y_pred)
print('RMSE between y_true (original) and y_pred: %.4f' % test_rmse)
pl.plot(y_true[:200], '-', y_pred[:200], '-o')
pl.legend(['Original', 'Predicted'], loc='upper left')
pl.show()
