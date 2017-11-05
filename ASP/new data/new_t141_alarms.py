"""
5/4/2017
1. generate logs based on threshold values
2. plot alarms

"""
import pandas as pd
import numpy as np
import pylab as pl

data = pd.read_pickle('new_t141.pkl')
names = ['L0310B', 'F1100', 'F0411']
path = "C:/Users/40204849/Desktop/Time Series Prediction/T141 Limits.csv"
limits = pd.read_csv(path, header=0, index_col=0)
logs = pd.DataFrame(index=data.index, columns=names)

for name in names:
    ll = limits.ix[name]['LL']
    low = limits.ix[name]['PL']
    high = limits.ix[name]['PH']
    hh = limits.ix[name]['HH']

    x = data[name]
    for i in range(len(x)):
        if x[i] < ll:
            logs.ix[logs.index[i]][name] = -2
        elif ll <= x[i] < low:
            logs.ix[logs.index[i]][name] = -1
        elif low <= x[i] < high:
            logs.ix[logs.index[i]][name] = 0
        elif high <= x[i] < hh:
            logs.ix[logs.index[i]][name] = 1
        else:
            logs.ix[logs.index[i]][name] = 2

a = logs['L0310B']
b = logs['F1100']
c = logs['F0411']

pl.figure(figsize=(12, 8))
ax = pl.gca()
ax.plot(a)
ax.plot(b)
pl.ylim([-2.5, 2.5])
ax.set_yticks(range(-2, 3, 1))
ax.set_yticklabels(['Low Low', 'Low', 'Normal', 'High', 'High High'])
pl.legend(['D131B - L0310B', 'R111 - F1100'], loc='best')
pl.savefig('new D131B R111.png', dpi=300)


pl.figure(figsize=(12, 8))
ax = pl.gca()
ax.plot(b)
ax.plot(c)
pl.ylim([-2.5, 2.5])
ax.set_yticks(range(-2, 3, 1))
ax.set_yticklabels(['Low Low', 'Low', 'Normal', 'High', 'High High'])
ax.legend(['R111 - F1100', 'T141 - F0411'], loc='best')
pl.savefig('new R111 T141.png', dpi=300)

pl.show()