"""
Training Period:        14/2/2016 12:01 AM to 4/3/2016 11:51 PM,    20 days, 2880 points
Testing Period 1:       5/3/2016 12:01 AM to 15/3/2016 11:51 PM,    11 days, 1584 points
Best k from fs_bayes_cv_1:         22

Daily re-train the model, check performance

1/11/2016: do scaling in the right way

"""

import pickle
import numpy as np
from sklearn import preprocessing
import pylab as pl
from sklearn.linear_model import BayesianRidge, LinearRegression

# p2 sampling data, 10 min
p2_sampling = pickle.load(open("p2_sampling.p", "rb"))

# selected columns
col_selected = pickle.load(open("selected_cols.p", "rb"))  # 30 features

# --------------------------------------------------
k_best = 22
selected_features = col_selected[:k_best]

outputs = p2_sampling[:, 149]
inputs = p2_sampling[:, selected_features]

train_size = 2880
slide_size = 144  # data size in 1 day
test_days = 11

Y_real = outputs[2880:4464]
Y_pred = np.zeros((test_days, slide_size))
for day in range(test_days):
    train_from = 2 + day * slide_size
    train_end = train_size + day * slide_size
    X, Y = inputs[train_from-1:train_end-1, :], outputs[train_from:train_end]

    test_from = 2880 + day * slide_size
    test_end = test_from + slide_size
    tX, tY = inputs[test_from-1:test_end-1, :], outputs[test_from:test_end]

    # scaling
    x_scaler = preprocessing.MinMaxScaler().fit(X)
    trainX = x_scaler.transform(X)
    testX = x_scaler.transform(tX)

    y_scaler = preprocessing.MinMaxScaler().fit(Y.reshape(-1, 1))  # Y is one column, need reshape
    trainY = y_scaler.transform(Y.reshape(-1, 1))
    testY = y_scaler.transform(tY.reshape(-1, 1))

    # regression
    model = LinearRegression().fit(trainX, trainY)
    testY_slide = model.predict(testX)
    temp = y_scaler.inverse_transform(testY_slide.reshape(-1, 1))
    Y_pred[day, :] = np.ravel(temp)

Y_est = np.ravel(Y_pred.reshape(1, -1))  # convert matrix to vector
test_rmse = np.sqrt(np.mean((Y_real - Y_est) ** 2))
print('The testing RMSE is: %.4f\n' % test_rmse)


# result display
pl.plot(Y_real, '-', Y_est, '-o', linewidth=2.0)
pl.show()

