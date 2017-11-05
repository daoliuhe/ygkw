"""
15/12/2016
Feature selection, copy from Nov/Regression
    a. filtered data is used
    b. 8 methods are used
"""

import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import \
    LinearRegression, Ridge, Lasso, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor
import warnings
from sklearn.feature_selection import f_regression, mutual_info_regression
from Dec.func import filter_data

# --------------------------------------------------
warnings.filterwarnings("ignore")

vib, inputs = filter_data()  # filtered data
s1 = preprocessing.MinMaxScaler()
vib_ = s1.fit_transform(vib)
s2 = preprocessing.MinMaxScaler()
inputs_ = s2.fit_transform(inputs)

# --------------------------------------------------
# feature selection
train_size = [3500, 4000, 4500, 5000, 5500]  # different lengths
mean_table = np.zeros((inputs.shape[1], len(train_size)))
horizon = 0  # forecasting horizon, e.g. 6-step (60 min) ahead
for kk in range(len(train_size)):
    len1 = train_size[kk]
    vibration = vib_[20:len1]
    inputs = inputs_[20-horizon:len1-horizon, :]

    # 1. linear regression
    lr = LinearRegression(normalize=True)
    lr.fit(inputs, vibration)
    w1 = np.abs(lr.coef_)

    # 2. ridge regression
    ridge = Ridge(alpha=1e-4)  # regularization parameter
    ridge.fit(inputs, vibration)
    w2 = np.abs(ridge.coef_)

    # 3.4.5 lasso
    lasso1 = Lasso(alpha=1e-3)
    lasso1.fit(inputs, vibration)
    w3 = np.abs(lasso1.coef_)

    lasso2 = LassoLarsIC(criterion='aic')
    lasso2.fit(inputs, vibration)
    w4 = np.abs(lasso2.coef_)

    lasso3 = LassoLarsIC(criterion='bic')
    lasso3.fit(inputs, vibration)
    w5 = np.abs(lasso3.coef_)

    # 6. random forest
    rf = RandomForestRegressor(n_estimators=20)
    rf.fit(inputs, vibration)
    w6 = rf.feature_importances_

    # 7. f_regression
    f, p = f_regression(inputs, vibration)
    w7 = np.nan_to_num(f)

    # 8. mi value
    mi_values = mutual_info_regression(inputs, vibration)
    w8 = mi_values

    # mean of 8 methods
    rank_table = np.vstack((w1, w2, w3, w4, w5, w6, w7, w8))
    # rank_table = np.vstack((w1, w2, w7, w8))  # remove forest, lasso*3
    rank_table = rank_table.T
    rank_scaler = preprocessing.MinMaxScaler()
    rank_scaled = rank_scaler.fit_transform(rank_table)
    mean_table[:, kk] = np.mean(rank_scaled, axis=1)

mean_list = np.mean(mean_table, axis=1)

# sort
ind = np.argsort(mean_list)     # small to big
ranked_cols = ind[::-1]         # big to small
pickle.dump(ranked_cols, open('ranked_cols.p', 'wb'))
print(ranked_cols)
print(mean_list[ranked_cols])
