"""
Training Period:        14/2/2016 12:01 AM to 4/3/2016 11:51 PM,    20 days, 2880 points
Testing Period 2:       10/3/2016 12:01 AM to 24/3/2016 11:51 PM,   15 days, 2160 points
Best k from fs_bayes_cv_1:         22

"""

import pickle
import numpy as np
from sklearn import preprocessing
import pylab as pl
from sklearn.linear_model import BayesianRidge, LinearRegression

# p2 sampling data, 10 min
p2_sampling = pickle.load(open("p2_sampling.p", "rb"))
scaler = preprocessing.MinMaxScaler()
p2_scaled = scaler.fit_transform(p2_sampling)

# selected columns
col_selected = pickle.load(open("selected_cols.p", "rb"))  # 30 features

# --------------------------------------------------
outputs = p2_scaled[:, 149]

k_best = 22
selected_features = col_selected[:k_best]
inputs = p2_scaled[:, selected_features]

train_size = 3600
trainX, trainY = inputs[1000:train_size - 1, :], outputs[1001:train_size]
test_from = 3600
test_end = 5760
testX, testY = inputs[test_from - 1:test_end-1, :], outputs[test_from:test_end]

# training
model = LinearRegression()
model.fit(trainX, trainY)
trainY_pred = model.predict(trainX)
train_rmse = np.sqrt(np.mean((trainY - trainY_pred) ** 2))
print('The training RMSE is: %.4f\n' % train_rmse)

# testing
testY_pred = model.predict(testX)
# test_rmse = np.sqrt(np.mean((testY - Y_pred)**2))
# print('The testing RMSE is: %.4f\n' % test_rmse)


# re-scaling
s1 = preprocessing.MinMaxScaler()
vibration = s1.fit_transform(p2_sampling[:, 149].reshape(-1, 1))
y = s1.inverse_transform(testY.reshape(-1, 1))
ybar = s1.inverse_transform(testY_pred.reshape(-1, 1))
test_rmse = np.sqrt(np.mean((y - ybar)**2))
print('The testing RMSE is: %.4f\n' % test_rmse)

# result display
# pl.figure(1)
# pl.plot(trainY, '-', trainY_pred, 'o', linewidth=2.0)
# pl.figure(2)
# pl.plot(testY[0:100], '-', Y_pred[0:100], 'o', linewidth=2.0)
# pl.figure(2)
# pl.plot(testY, '-', Y_pred, '-o', linewidth=2.0)
# pl.figure(2)
# pl.plot(testY[0:50], '-', Y_pred[0:50], '-o', linewidth=2.0)
# pl.plot(np.abs(model.coef_))

# pl.show()
pl.plot(y, '-', ybar, '-o', linewidth=2.0)
pl.show()

