"""
Use t-1 and t-2 to predict t

High correlation between inputs as same variable but different lags are selected

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

k_best = 20
selected_features = col_selected[:k_best]
inputs = p2_scaled[:, selected_features]

train_size = 3600
slide_size = 144  # points in 1 day
test_days = 15

testY_actual = outputs[3600:5760]
testY_pred = np.zeros((test_days, slide_size))
for day in range(test_days):
    train_from = 1000 + day * slide_size
    train_end = train_size + day * slide_size
    X1 = inputs[train_from-1:train_end-1, :]
    X2 = inputs[train_from-2:train_end-2, :]
    trainX = np.concatenate((X1, X2), axis=1)
    trainY = outputs[train_from:train_end]

    test_from = 3600 + day * slide_size
    test_end = test_from + slide_size
    TX1 = inputs[test_from-1:test_end-1, :]
    TX2 = inputs[test_from-2:test_end-2, :]
    testX = np.concatenate((TX1, TX2), axis=1)
    testY = outputs[test_from:test_end]

    # training
    model = LinearRegression()
    model.fit(trainX, trainY)
    # trainY_pred = model.predict(trainX)
    # train_rmse = np.sqrt(np.mean((trainY - trainY_pred) ** 2))
    # print('The training RMSE is: %.4f\n' % train_rmse)

    # testing
    testY_slide = model.predict(testX)
    testY_pred[day, :] = testY_slide

test_predict = np.ravel(testY_pred.reshape(1, -1))
# test_rmse = np.sqrt(np.mean((Y_real - Y_est)**2))
# print('The testing RMSE is: %.4f\n' % test_rmse)

# re-scaling
s1 = preprocessing.MinMaxScaler()
vibration = s1.fit_transform(p2_sampling[:, 149].reshape(-1, 1))
y = s1.inverse_transform(testY_actual.reshape(-1, 1))
ybar = s1.inverse_transform(test_predict.reshape(-1, 1))
test_rmse = np.sqrt(np.mean((y - ybar)**2))
print('The testing RMSE is: %.4f\n' % test_rmse)

# result display
# pl.plot(Y_real, '-', Y_est, '-o', linewidth=2.0)
pl.plot(y, '-', ybar, '-o', linewidth=2.0)
pl.show()
