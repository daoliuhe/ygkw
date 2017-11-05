"""
Using cross validation to determine the number of features
    a. number of selected features
    b. data split ratio

selected_cols.p generated from fs_length

30/10/2016: best k: 20 >>> 22
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
col_selected = pickle.load(open("selected_cols.p", "rb"))

# --------------------------------------------------
len1 = 2880
outputs = p2_scaled[:len1, 149]

kbest = range(1, 51)
rmse = []
for k in kbest:
    selected_features = col_selected[:k]
    inputs = p2_scaled[:len1, selected_features]

    # data split
    split_ratio = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    len2 = len(split_ratio)
    temp = []

    for ii in range(len2):
        train_size = round(len1 * split_ratio[ii])
        trainX, trainY = inputs[1:train_size - 1, :], outputs[2:train_size]
        testX, testY = inputs[train_size - 1:-1, :], outputs[train_size:]

        # training
        model = BayesianRidge()
        model.fit(trainX, trainY)

        # testing
        testY_pred = model.predict(testX)
        test_rmse = np.sqrt(np.mean((testY - testY_pred) ** 2))
        temp.append(test_rmse)

    rmse.append(np.mean(temp))

pl.plot(kbest, rmse, 'r-o', linewidth=2.0)
pl.xlabel('Number of selected features')
pl.ylabel('RMSE on validation set')

pl.show()

