"""
8/12/2016
    1. check correlation between vibration and lagged inputs
    2. lfilter makes increasing trend not obvious
    3. remove features with low variances
    4. remove features with low correlation
    5. MinMaxScaler does not affect pearson r

22/12/2016
    6. pcc varies very litter wrt lag
    7. different variables have different lag-pcc curves
"""

import pickle
from User.functions import slice_data, pcc, variance
import numpy as np
import pylab as pl
from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import normalized_mutual_info_score


data_sampled = pickle.load(open("../Data/sampled2016.p", "rb"))
date_sampled = pickle.load(open("../Data/date2016.p", "rb"))
date_start, date_end = '01/Feb/16 00:10:00', '01/Jun/16 00:10:00'
data = slice_data(date_sampled, data_sampled, date_start, date_end)
vib = data[:, 41]  # XI392-04A

col_ind = pickle.load(open("../Nov/Regression/col_index_new.p", "rb"))
inputs = data[:, col_ind]
print(inputs.shape)  # 62

# remove features when r < 0.1
r = pcc(inputs, vib)
r_ind = r >= 0.1
inputs1 = inputs[:, r_ind]
col_ind1 = np.array(col_ind)[r_ind]
print(inputs1.shape)  # 49

# remove features when var < 0.005
# do scaling first
sx = MinMaxScaler()
inputs2 = sx.fit_transform(inputs1)
var = variance(inputs2)
var_ind = var >= 0.005
inputs3 = inputs1[:, var_ind]
col_ind2 = col_ind1[var_ind]
print(inputs3.shape)  # 48

# correlation (y, x_lag)
start = 100
end = len(vib)
lags = list(range(1, 21))
corr = np.zeros((len(lags), len(col_ind2)))
y = vib[start:end]
for i in range(len(lags)):
    lag = lags[i]
    for j in range(len(col_ind2)):
        x_col = inputs3[:, j]
        x = x_col[start-lag:end-lag]
        corr[i, j] = np.abs(pearsonr(x, y)[0])

for j in range(len(col_ind2)):
    if j < 10:
        pl.figure(j)
        pl.plot(corr[:, j])

pl.show()