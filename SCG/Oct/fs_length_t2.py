"""
Using different lengths of X/y for feature selection

27/10/2016: Select variables at time t-2

"""

import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import \
    LinearRegression, Ridge, Lasso, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.metrics import normalized_mutual_info_score
from sklearn.feature_selection import f_regression


# original variable list
col_index = pickle.load(open("col_index.p", "rb"))

# p2 sampled data
p2_sampling = pickle.load(open("p2_sampling.p", "rb"))

scaler = preprocessing.MinMaxScaler()  # scale to [0, 1]
p2_scaled = scaler.fit_transform(p2_sampling)

# remove variables if variance < 0.01
col_index_new = []
for i in range(0, len(col_index)):
    temp = np.var(p2_scaled[:, col_index[i]]) >= 0.01
    if temp:
        if col_index[i] != 149:                         # ignore vibration itself
            col_index_new.append(col_index[i])

# --------------------------------------------------
# feature selection
train_size = [3500, 4000, 4500, 5000, 5500]  # different lengths
mean_table = np.zeros((len(col_index_new), len(train_size)))
for kk in range(len(train_size)):
    len1 = train_size[kk]
    vibration = p2_scaled[10:len1, 149]  # y
    # inputs = p2_scaled[9:len1 - 1, col_index_new]  # variables at time t-1
    inputs = p2_scaled[8:len1 - 2, col_index_new]  # variables at time t-2

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

    # 7. Cross correlation
    corrs = np.zeros(len(col_index_new))
    for i in range(0, len(col_index_new)):
        corrs[i] = pearsonr(inputs[:, i], vibration)[0]
    w7 = np.abs(corrs)

    # 8. mi value
    mi_values = np.zeros(len(col_index_new))
    for i in range(0, len(col_index_new)):
        mi_values[i] = normalized_mutual_info_score(inputs[:, i], vibration)
    w8 = mi_values

    # mean of 8 methods
    rank_table = np.vstack((w1, w2, w3, w4, w5, w6, w7, w8))
    rank_table = rank_table.T
    rank_scaler = preprocessing.MinMaxScaler()
    rank_scaled = rank_scaler.fit_transform(rank_table)
    mean_table[:, kk] = np.mean(rank_scaled, axis=1)

mean_list = np.mean(mean_table, axis=1)

# sort
ind = np.argsort(mean_list)  # small to big order
ind = ind[::-1]

col_index_array = np.array(col_index_new)
selected_cols = col_index_array[ind[:50]]

pickle.dump(selected_cols, open('selected_cols.p', 'wb'))
