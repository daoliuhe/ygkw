"""
Consider y_t-1 as a forecast of y_t

"""

import pickle
import numpy as np
from sklearn import preprocessing
import pylab as pl
from sklearn.linear_model import BayesianRidge, LinearRegression

# p2 sampling data, 10 min
p2_sampling = pickle.load(open("p2_sampling.p", "rb"))

# test period 1
# outputs = p2_sampling[:, 149]
# Y_real = outputs[2880:4464]
# Y_pred = outputs[2880-1:4464-1]
#
# test_rmse = np.sqrt(np.mean((Y_real - Y_pred)**2))
# print('The testing RMSE is: %.4f\n' % test_rmse)


# test period 2
outputs = p2_sampling[:, 149]
testY_actual = outputs[3600:5760]
testY_pred = outputs[3600-1:5760-1]

test_rmse = np.sqrt(np.mean((testY_actual - testY_pred)**2))
print('The testing RMSE is: %.4f\n' % test_rmse)

