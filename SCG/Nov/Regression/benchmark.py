"""
7/11/2016
Benchmark model for multi-step prediction
            y_t = y_t-6

RMSE 2016:   0.2508
RMSE 2014:   0.9277

"""

import pickle
import numpy as np
import pylab as pl
from User.functions import slice_data

# case 2016
data_sampled = pickle.load(open("../Data/sampled2016.p", "rb"))
date_sampled = pickle.load(open("../Data/date2016.p", "rb"))
date_start = '14/Feb/16 00:10:00'
date_end = '25/Mar/16 00:10:00'

# case 2014
# data_sampled = pickle.load(open("../Data/sampled2014.p", "rb"))
# date_sampled = pickle.load(open("../Data/date2014.p", "rb"))
# date_start = '05/Mar/14 00:10:00'
# date_end = '14/Apr/14 00:10:00'

data_sliced = slice_data(date_sampled, data_sampled, date_start, date_end)
outputs = data_sliced[:, 41]
Y_real = outputs[3600:5760]
Y_est = outputs[3600-6:5760-6]

test_rmse = np.sqrt(np.mean((Y_real - Y_est) ** 2))
print('The testing RMSE is: %.4f\n' % test_rmse)

pl.plot(Y_real, '-s', Y_est, '-o', linewidth=2.0)
pl.show()