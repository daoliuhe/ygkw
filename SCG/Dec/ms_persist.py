"""
19/12/2016
Persistence for 2016: RMSE=0.1987

4/1/2017
Persistence for 2014: rmse=0.5399
"""

import pickle
from User.functions import slice_data, rmse
import pylab as pl
import numpy as np
import time

# year 2016
# data_sampled = pickle.load(open("../Data/sampled2016.p", "rb"))
# date_sampled = pickle.load(open("../Data/date2016.p", "rb"))
# date_start, date_end = '14/Feb/16 00:10:00', '25/Mar/16 00:10:00'
# data = slice_data(date_sampled, data_sampled, date_start, date_end)

# year 2014
data_sampled = pickle.load(open("../Data/sampled2014.p", "rb"))
date_sampled = pickle.load(open("../Data/date2014.p", "rb"))
date_start, date_end = '05/Mar/14 00:10:00', '14/Apr/14 00:10:00'
data = slice_data(date_sampled, data_sampled, date_start, date_end)
vibration = data[:, 41]

train_size = 3600
testY = vibration[3600:]
testY_pred = np.zeros(len(testY))
horizon = 6
n_hours = int(len(testY) / horizon)  # for hourly update
Y = vibration[3599:]
tic = time.time()
for hour in range(n_hours):
    start = hour * horizon
    testY_pred[start:start+horizon] = np.ones(horizon) * Y[start]
toc = time.time()
print("Running time: %.4f" % (toc - tic))
test_rmse = rmse(testY, testY_pred)
print('Testing RMSE: %.4f' % test_rmse)
pl.plot(testY, '-', testY_pred, '-o')
# pl.show()