"""
5/7/2017
T141 case

H_0: ['L0310B', 'F1100'] do not Granger-cause L0410
Conclusion: fail to reject H_0 at 5.00% significance level

H_0: ['L0310B'] do not Granger-cause L0410
Conclusion: reject H_0 at 5.00% significance level

H_0: ['F1100'] do not Granger-cause L0410
Conclusion: fail to reject H_0 at 5.00% significance level

H_0: ['F1100'] do not Granger-cause L0310B
Conclusion: fail to reject H_0 at 5.00% significance level

H_0: ['L0310B'] do not Granger-cause F1100
Conclusion: fail to reject H_0 at 5.00% significance level

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
import pylab as pl

data = pd.read_pickle('t141.pkl')
names = ['L0310B', 'F1100', 'L0410']
df = data[names].astype(float)
df = df['2013-07-10 00:00:00':'2013-07-15 23:59:00'].resample('10T').mean()
mdata = np.log(df).diff().dropna()
# pl.plot(mdata[names[0]])
# pl.plot(mdata[names[1]])
# pl.plot(mdata[names[2]])

model = VAR(mdata)
# model.select_order(10)
result = model.fit(8)

result.test_causality('F1100', ['L0310B'], kind='f')

# result.plot()
# pl.show()