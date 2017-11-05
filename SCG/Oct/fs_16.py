"""
Using different lengths of X/y for feature selection

30/10/2016
LASSO AIC/BIC: set max_iter=100 to avoid convergence warning

4/11/2016
    a. new data not good, back to old one
    b. do not show warnings
    c. replace PCC and MI with f_regression and mutual_info_regression
    d. make step reasonable and convincing

"""

import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import \
    LinearRegression, Ridge, Lasso, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor
import warnings
from scipy.stats import pearsonr
from sklearn.metrics import normalized_mutual_info_score
from sklearn.feature_selection import f_regression, mutual_info_regression

# not show warnings
warnings.filterwarnings("ignore")

# original variable list
col_index = pickle.load(open("col_index.p", "rb"))

# p2 sampled data
p2_sampling = pickle.load(open("p2_sampling.p", "rb"))

scaler = preprocessing.MinMaxScaler()  # scale to [0, 1]
p2_scaled = scaler.fit_transform(p2_sampling)

# remove XI392-04A (149), XI392-04B (150), XI392-12 (165)
col_index_new = []
for index in col_index:
    temp = np.var(p2_scaled[:, index]) >= 0.01
    if temp:
        if index == 149:
            pass
        elif index == 150:
            pass
        elif index == 165:
            pass
        else:
            col_index_new.append(index)

# --------------------------------------------------
# feature selection
train_size = 5760
split_ratio = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
mean_table = np.zeros((len(col_index_new), len(split_ratio)))
for kk in range(len(split_ratio)):
    len1 = round(train_size * split_ratio[kk])
    vibration = p2_scaled[10:len1, 149]
    inputs = p2_scaled[9:len1-1, col_index_new]  # x_t-1 ---> y_t

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

    lasso2 = LassoLarsIC(criterion='aic', max_iter=100)
    lasso2.fit(inputs, vibration)
    w4 = np.abs(lasso2.coef_)

    lasso3 = LassoLarsIC(criterion='bic', max_iter=100)
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
ind = np.argsort(mean_list)  # small to big order
ind = ind[::-1]

col_index_array = np.array(col_index_new)
selected_cols = col_index_array[ind[:50]]

pickle.dump(selected_cols, open('selected_cols.p', 'wb'))
