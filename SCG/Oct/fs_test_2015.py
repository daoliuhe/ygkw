"""
Training Period:    14/2/2016 12:01 AM to 24/3/2016 11:51 PM, 40 days, 5760 points
Testing 2015:       10/1/2015 12:01 AM to 19/1/2015 11:51 PM, 10 days, 1440 points
Best k from fs_bayes_cv_1:         22


31/10/2016: use 2016 to predict 2015
    a. data series shape can be predicted
    b. but a clear gap between actual and predict
"""

import pickle
import numpy as np
from sklearn import preprocessing
import pylab as pl
from sklearn.linear_model import BayesianRidge, LinearRegression

# p2 sampling data, 10 min
p2_sampling = pickle.load(open("p2_sampling.p", "rb"))
# p2 2015 sampling data
p2_2015 = pickle.load(open("p2_2015_sampling.p", "rb"))

scaler = preprocessing.MinMaxScaler()
p2_scaled = scaler.fit_transform(p2_sampling)
p2_2015_scaled = scaler.fit_transform(p2_2015)

# selected columns
col_selected = pickle.load(open("selected_cols.p", "rb"))  # 30 features

# --------------------------------------------------
k_best = 22
selected_features = col_selected[:k_best]

outputs = p2_scaled[:, 149]
inputs = p2_scaled[:, selected_features]
train_size = 5760
trainX, trainY = inputs[1000:train_size - 1, :], outputs[1001:train_size]

outputs_2015 = p2_2015_scaled[:, 149]
inputs_2015 = p2_2015_scaled[:, selected_features]
test_size = 1440
testX, testY = inputs_2015[0:test_size, :], outputs_2015[1:test_size+1]

# training
model = LinearRegression()
model.fit(trainX, trainY)
trainY_pred = model.predict(trainX)
train_rmse = np.sqrt(np.mean((trainY - trainY_pred) ** 2))
print('The training RMSE is: %.4f\n' % train_rmse)

# testing
testY_pred = model.predict(testX)
test_rmse = np.sqrt(np.mean((testY - testY_pred)**2))
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
pl.plot(testY, '-', testY_pred, '-o', linewidth=2.0)
pl.show()

