"""
Perform re-sampling on P2

"""

import pickle
import numpy as np

# original datetime list
datetimes = pickle.load(open("datetimes.p", "rb"))

# 25/10/2016 extend training period
train_start_row = datetimes['14/2/2016 12:01 AM']   # row 1
test_end_row = datetimes['25/3/2016 12:00 AM']      # row 57600

# p2 data loading
p2_data = pickle.load(open("p2_data.p", "rb"))
p2_data = np.nan_to_num(p2_data)  # convert nan to 0
p2_period = p2_data[train_start_row:test_end_row + 1, :]

# re-sampling: average 10 points
m = round(p2_period.shape[0]/10)
n = p2_period.shape[1]
p2_sampling = np.zeros((m, n))

for j in range(0, n):
    for i in range(0, m):
        p2_sampling[i, j] = np.mean(p2_period[10*i:10*(i+1), j])

pickle.dump(p2_sampling, open("p2_sampling.p", "wb"))
