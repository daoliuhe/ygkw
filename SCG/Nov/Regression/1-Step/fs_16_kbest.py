"""
Using cross validation to determine the number of features
    a. number of selected features
    b. data split ratio

2/11/2016:      best k: 37
4/11/2016:      best k: 7
"""

import pickle
import numpy as np
from sklearn import preprocessing
import pylab as pl
from sklearn.linear_model import LinearRegression, BayesianRidge
from User.functions import slice_data

# data in a range
data_sampled = pickle.load(open("../Data/sampled2016.p", "rb"))
date_sampled = pickle.load(open("../Data/date2016.p", "rb"))
date_start = '14/Feb/16 00:10:00'
date_end = '25/Mar/16 00:10:00'

data_sliced = slice_data(date_sampled, data_sampled, date_start, date_end)
scaler = preprocessing.MinMaxScaler()
data_scaled = scaler.fit_transform(data_sliced)

# selected columns
col_selected = pickle.load(open("selected_cols.p", "rb"))

# --------------------------------------------------
len1 = 3600
outputs = data_scaled[:len1, 41]

kbest = range(1, len(col_selected))
rmse = []
for k in kbest:
    selected_features = col_selected[:k]
    inputs = data_scaled[:len1, selected_features]

    # data split
    split_ratio = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    len2 = len(split_ratio)
    temp = []

    for ii in range(len2):
        train_size = round(len1 * split_ratio[ii])
        trainX, trainY = inputs[1:train_size - 1, :], outputs[2:train_size]
        testX, testY = inputs[train_size - 1:-1, :], outputs[train_size:]

        # regression
        model = LinearRegression().fit(trainX, trainY)
        testY_pred = model.predict(testX)
        test_rmse = np.sqrt(np.mean((testY - testY_pred) ** 2))
        temp.append(test_rmse)

    rmse.append(np.mean(temp))

best_k = kbest[rmse.index(min(rmse))]
print('The best k is: %s' % best_k)
pl.plot(kbest, rmse, 'r-o', linewidth=2.0)
pl.xlabel('Number of selected features')
pl.ylabel('RMSE on validation set')

# pl.show()