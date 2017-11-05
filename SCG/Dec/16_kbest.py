"""
15/12/2016
CV to select k

"""

import pickle
import numpy as np
from sklearn import preprocessing
import pylab as pl
from sklearn.linear_model import LinearRegression, BayesianRidge
from Dec.func import filter_data
from sklearn.svm import SVR

vib, inputs = filter_data()  # filtered data
s1 = preprocessing.MinMaxScaler()
vib_ = s1.fit_transform(vib)
s2 = preprocessing.MinMaxScaler()
inputs_ = s2.fit_transform(inputs)

# ranked columns
ranked_cols = pickle.load(open("ranked_cols.p", "rb"))

# --------------------------------------------------
len1 = 3600
outputs = vib_[:len1]

horizon = 6  # forecasting horizon, e.g. 6-step (60 min) ahead
kbest = range(1, len(ranked_cols))
rmse = []

for k in kbest:
    selected_features = ranked_cols[:k]
    inputs = inputs_[:len1, selected_features]

    # data split
    split_ratio = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    len2 = len(split_ratio)
    temp = []

    for ii in range(len2):
        train_size = round(len1 * split_ratio[ii])
        trainX, trainY = inputs[1:train_size-horizon, :], outputs[1+horizon:train_size]
        testX, testY = inputs[train_size-horizon:-horizon, :], outputs[train_size:]

        # regression
        # model = SVR()
        model = LinearRegression()
        model.fit(trainX, trainY)
        testY_pred = model.predict(testX)
        test_rmse = np.sqrt(np.mean((testY - testY_pred) ** 2))
        temp.append(test_rmse)

    rmse.append(np.mean(temp))

best_k = kbest[rmse.index(min(rmse))]
print('The best k is: %s' % best_k)
print('Min RMSE: %s' % min(rmse))
pl.plot(kbest, rmse, '-o', linewidth=2.0)
pl.xlabel('Number of selected features')
pl.ylabel('RMSE on validation set')

# pl.show()
