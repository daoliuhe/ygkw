"""
14/12/2016
Use mlxtend to perform feature selection

1. Linear
    a. forward: k=7, [32, 34, 4, 8, 57, 44, 45]
    b. backward: k=13, [0, 5, 7, 40, 41, 39, 42, 12, 43, 19, 20, 53, 55]
    c. forward float: k=15, [32, 2, 3, 4, 5, 36, 6, 34, 8, 10, 11, 45, 14, 57, 28]
    d. backward float: k=20, [0, 34, 35, 36, 5, 6, 7, 39, 40,
    41, 42, 12, 13, 43, 45, 19, 20, 52, 53, 55]

2. Ridge
    a. forward: k=19, [1, 2, 3, 4, 5, 6, 8, 10, 16, 29,
    30, 32, 34, 36, 44, 45, 47, 51, 57]
    b. backward: k=18, [0, 34, 35, 4, 5, 39, 40, 41, 10,
    42, 12, 14, 47, 50, 53, 54, 26, 59]
    c. forward float: k=25, [0, 1, 2, 3, 4, 5, 6, 9, 10, 11,
    14, 18, 19, 21, 25, 29, 30, 32, 34, 36, 45, 47, 48, 57, 61]
    d. backward float: k=21, [0, 4, 5, 10, 12, 14, 23, 27, 30,
     34, 35, 36, 39, 40, 41, 42, 46, 47, 50, 54, 59]

3. SVR
    a. forward: k=19, [0, 5, 8, 10, 11, 12, 14, 17, 20,
    23, 28, 32, 34, 35, 38, 41, 42, 57, 61]
    b. backward: k=19, [0, 32, 34, 3, 5, 39, 8, 40, 10, 41,
    42, 14, 45, 17, 20, 54, 55, 57, 61]
    c. forward float: k=26, [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14,
    15, 16, 17, 20, 33, 34, 35, 38, 41, 42, 49, 55, 57, 59, 61]
    d. backward float: k=19, [0, 32, 2, 34, 4, 5, 35, 7, 38, 39,
    10, 40, 41, 42, 14, 17, 57, 28, 61]

"""

import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Dec.func import remove_low_r_var, filter_data
from sklearn.linear_model import LinearRegression, Ridge
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import pylab as pl
import pandas as pd
from sklearn.svm import SVR
from User.functions import find_name


vib, inputs = filter_data()
s1 = MinMaxScaler()
vib_ = s1.fit_transform(vib)
s2 = MinMaxScaler()
inputs_ = s2.fit_transform(inputs)

lr = LinearRegression()
# rd = Ridge()
# svr = SVR()
sfs = SFS(lr,
          k_features=13,
          forward=False,
          floating=False,
          scoring='neg_mean_squared_error',
          cv=5)

sfs = sfs.fit(inputs_, vib_)
# result = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
# result.to_csv('result.csv', encoding='utf-8')
# fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
# pl.grid()
# pl.show()
k_index = sfs.k_feature_idx_  # tuple
col_ind = pickle.load(open("../Nov/Regression/col_index_new.p", "rb"))  # list
# print(list(np.array(col_ind)[list(k_index)]))
selected_cols = list(np.array(col_ind)[list(k_index)])
print('\n')
find_name(selected_cols)  # find selected variable names
