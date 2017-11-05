"""
7/11/2016
multi-step prediction

"""

import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import \
    LinearRegression, Ridge, Lasso, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor
import warnings
from User.functions import slice_data
from sklearn.feature_selection import f_regression, mutual_info_regression

# --------------------------------------------------
warnings.filterwarnings("ignore")

# original variable list
col_index = pickle.load(open("col_index.p", "rb"))

# remove XI392-04A, XI392-04B
col_index_new = []
for index in col_index:
    if index == 41:  # XI392-04A
        pass
    elif index == 42:  # XI392-04B
        pass
    else:
        col_index_new.append(index)

# --------------------------------------------------
# data in a range
data_sampled = pickle.load(open("C:/Users/40204849/PycharmProjects/SCG/Data/sampled2016.p", "rb"))
date_sampled = pickle.load(open("C:/Users/40204849/PycharmProjects/SCG/Data/date2016.p", "rb"))
date_start = '14/Feb/16 00:10:00'
date_end = '25/Mar/16 00:10:00'

data_sliced = slice_data(date_sampled, data_sampled, date_start, date_end)
scaler = preprocessing.MinMaxScaler()
data_scaled = scaler.fit_transform(data_sliced)

# --------------------------------------------------
# feature selection
train_size = [3500, 4000, 4500, 5000, 5500]  # different lengths
mean_table = np.zeros((len(col_index_new), len(train_size)))
horizon = 6  # forecasting horizon, e.g. 6-step (60 min) ahead
for kk in range(len(train_size)):
    len1 = train_size[kk]
    vibration = data_scaled[20:len1, 41]
    inputs = data_scaled[20-horizon:len1-horizon, col_index_new]

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
    rank_table = rank_table.T
    rank_scaler = preprocessing.MinMaxScaler()
    rank_scaled = rank_scaler.fit_transform(rank_table)
    mean_table[:, kk] = np.mean(rank_scaled, axis=1)

mean_list = np.mean(mean_table, axis=1)

# sort
ind = np.argsort(mean_list)     # small to big
ind = ind[::-1]                 # big to small

col_index_array = np.array(col_index_new)
selected_cols = col_index_array[ind]

pickle.dump(selected_cols, open('selected_cols.p', 'wb'))

