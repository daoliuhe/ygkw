"""
5/7/2017
Granger test, C145 case

H_0: ['L0330'] do not Granger-cause T0451A
Conclusion: fail to reject H_0 at 5.00% significance level
Remark: L0330 NOT cause T0451A

H_0: ['P0457'] do not Granger-cause T0451A
Conclusion: reject H_0 at 5.00% significance level
Remark: P0457 cause T0451A

H_0: ['L0330'] do not Granger-cause P0457
Conclusion: fail to reject H_0 at 5.00% significance level
Remark: L0330 NOT cause P0457

H_0: ['L0330', 'P0457'] do not Granger-cause T0451A
Conclusion: reject H_0 at 5.00% significance level
Remark: ['L0330', 'P0457'] cause T0451A

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
import pylab as pl

data = pd.read_pickle('c145.pkl')
df = data['2014-11-15 00:00:00':'2014-11-22 23:59:00'].resample('10T').mean()
names = ['L0330', 'P0457', 'T0451A']
mdata = np.log(df[names]).diff().dropna()
# pl.plot(mdata[names[0]])
# pl.plot(mdata[names[1]])
# pl.plot(mdata[names[2]])

model = VAR(mdata)
# model.select_order(10)
result = model.fit(2)

# result.test_causality('T0451A', ['L0330','P0457'], kind='f')
result.test_causality('T0451A', ['L0330'], kind='f')

# result.plot()
pl.show()