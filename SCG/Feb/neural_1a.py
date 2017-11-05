"""
7/2/2017
Neural network with external variables
Remark: not good, not help

"""

from User.functions import rmse, read_data, prepare_data_, filter_data_
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

rank_cols = pickle.load(open('../Jan/ranking_cols.p', 'rb'))

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
X_input = data[:, rank_cols[:2]]
trainX, trainY, testX, testY = prepare_data_(vib_filtered, X_input, train_size, n_y_input)

model = MLPRegressor(hidden_layer_sizes=(10, ), solver='lbfgs')
model.fit(trainX, trainY)
y_pred = model.predict(testX)

test_rmse = rmse(y_true, y_pred)
print('RMSE between y_true (original) and y_pred: %.4f' % test_rmse)