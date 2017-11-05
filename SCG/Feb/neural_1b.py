"""
7/2/2017 Normalization
with StandardScaler: 1.23, 1.62, 1.56
with np.max: 0.1555, 0.3277, 0.3596
with minmaxscaler: 0.1341, 0.3072, 0.2209
remark: normalization makes situation worse, in this case
"""

from User.functions import rmse, read_data, prepare_data, filter_data_
from sklearn.neural_network import MLPRegressor
import pylab as pl
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression

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

vib = filter_data_(vibration, filter_size)
# vib_filtered = vib / np.max(vib)
scaler = MinMaxScaler()
vib_filtered = scaler.fit_transform(vib.reshape(-1, 1)).reshape(len(vib))
trainX, trainY, testX, testY = prepare_data(vib_filtered, train_size, n_y_input)
model = MLPRegressor(hidden_layer_sizes=(10, ), solver='lbfgs')
# model = LinearRegression()
model.fit(trainX, trainY)
# y_pred = model.predict(testX) * np.max(vib)
predict = model.predict(testX)
y_pred = scaler.inverse_transform(predict).reshape(len(predict))
test_rmse = rmse(y_true, y_pred)
print('RMSE between y_true (original) and y_pred: %.4f' % test_rmse)
pl.figure(figsize=(15, 10))
pl.plot(y_true, '-', y_pred, '-o')
pl.legend(['Original', 'Predicted'], loc='upper left')
pl.show()