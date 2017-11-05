"""
7/2/2017 Neural network ensemble
similar results produced
"""

from User.functions import rmse, read_data, prepare_data, filter_data_

from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler

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

n_models = 3
preds = np.zeros((n_models, len(y_true)))
for i in range(n_models):
    model = MLPRegressor(hidden_layer_sizes=(10,), solver='lbfgs')
    model.fit(trainX, trainY)
    preds[i, :] = model.predict(testX)

y_pred = np.mean(preds, axis=0)

test_rmse = rmse(y_true, y_pred)
print('RMSE between y_true (original) and y_pred: %.4f' % test_rmse)