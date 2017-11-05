"""
21/12/2016
Fine-tune NN model to improve accuarcy
    a. normalization brings degraded accuracy
    b. more hidden nodes have better accuracy
    c. adam is best, relu is best
    d. no. of hidden layers
        - (100, )           0.1887
        - (100, 100)        0.1870
        - (100, 10)         1.0170
        - (100, 10, 10)     0.9848
        - (100, 100, 100)   0.1965

"""

import pickle
from User.functions import slice_data, rmse
import pylab as pl
from sklearn.svm import SVR
import numpy as np
from Dec.func import prepare_data, iterative_forecast, print_progress
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

data_sampled = pickle.load(open("../Data/sampled2016.p", "rb"))
date_sampled = pickle.load(open("../Data/date2016.p", "rb"))
date_start, date_end = '14/Feb/16 00:10:00', '25/Mar/16 00:10:00'
data = slice_data(date_sampled, data_sampled, date_start, date_end)
vibration = data[:, 41]

train_size = 3600
trainX, trainY, testX, testY = prepare_data(vibration, train_size)

horizon = 6
n_features = 10
n_hours = int(len(testY) / horizon)  # for hourly update
X = np.vstack((trainX, testX))
Y = np.hstack((trainY, testY))
train_len = len(trainY)
testY_pred = np.zeros(len(testY))
tic = time.time()

# without norlization
for hour in range(n_hours):
    start = hour * horizon + 0
    end = hour * horizon + train_len
    trainX_new = X[start:end, :]
    trainY_new = Y[start:end]
    model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), early_stopping=True)
    model.fit(trainX_new, trainY_new)  # update model hourly

    testX_batch = testX[start:start+horizon, :]
    batch_forecast = iterative_forecast(model, testX_batch, horizon, n_features)
    testY_pred[start:start+horizon] = batch_forecast
    print_progress(hour, n_hours, 25)
# ----------------------------------

toc = time.time()
print("Running time: %.4f" % (toc - tic))
test_rmse = rmse(testY, testY_pred)
print('Testing RMSE: %.4f' % test_rmse)
pl.plot(testY, '-', testY_pred, '-o')
pl.show()

