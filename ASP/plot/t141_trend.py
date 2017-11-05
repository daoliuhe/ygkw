"""
31/7/2017
Plot all tags in T141 case
    D131B, R111, T141
"""

import pandas as pd
import pylab as pl

data = pd.read_pickle('../pkl/t141.pkl')
tags = ['L0310B', 'L0311B', 'P0310B', 'T0310B', 'F1100',
        'P1100', 'X1100', 'F0411', 'F0413', 'L0410',
        'P0410', 'P0411', 'T0410', 'T0411', 'T0412',
        'T0414', 'T0415', 'T0416']

df = data[tags]['2013-07-15 08:20:00':'2013-07-15 23:30:00']
df /= df.max(axis=0)

n = len(tags) // 3
for i in range(n):
    names = tags[3*i:3+3*i]
    f, axarr = pl.subplots(3, sharex=True, figsize=(14, 10))
    axarr[0].plot(df[names[0]])
    axarr[0].set_title(names[0])
    axarr[1].plot(df[names[1]])
    axarr[1].set_title(names[1])
    axarr[2].plot(df[names[2]])
    axarr[2].set_title(names[2])

pl.show()