"""
27/10/2016: Find datetime for number index

"""

import numpy as np

data = [line.rstrip('\n') for line in open('datetime.txt')]
data = np.array(data[1:])  # skip first row
ind = range(0, len(data), 10)  # sampling: 10-min data

# test period 1
# date1 = data[ind]
# date2 = date1[2880:]

# ind2 = range(0, len(date2), 200)
# date3 = date2[ind2]

# test period 2: [3600, 5760]
date1 = data[ind]  # 10-min data
date2 = date1[3600:]  # testing part
ind3 = range(0, len(date2), 500)  # axis resolution: 500
date4 = date2[ind3]   # datetime for axis
