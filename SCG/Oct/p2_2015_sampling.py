"""
Perform re-sampling on p2_20150110.p

Data period: 10jan15_19feb15

"""

import pickle
import numpy as np

# data range
train_start_row = 1
test_end_row = 57600

# p2 2015 loading
p2_2015 = pickle.load(open("p2_20150110.p", "rb"))
p2_period = p2_2015[train_start_row:test_end_row + 1, :]

# re-sampling: average 10 points
m = round(p2_period.shape[0]/10)
n = p2_period.shape[1]
p2_sampling = np.zeros((m, n))

for j in range(0, n):
    for i in range(0, m):
        p2_sampling[i, j] = np.mean(p2_period[10*i:10*(i+1), j])

pickle.dump(p2_sampling, open("p2_2015_sampling.p", "wb"))
