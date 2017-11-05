"""
31/7/2017
Plot all associated tags in C145 case
    D133, F145, C145
"""

import pandas as pd
import pylab as pl

data = pd.read_pickle('../pkl/c145.pkl')
tags = ['L0330', 'P0457', 'P0451', 'P0452',
        'P0452A', 'P0453', 'P0453A', 'P0455', 'P0465',
        'T0413', 'T0450A', 'T0451', 'T0451A', 'T0452',
        'T0452A', 'T0453', 'T0458', 'T0459', 'T0460',
        'X0451A', 'X0451B', 'X0451C',
        'X0451D', 'X0451E', 'X0451F']  # 25 tags


df = data[tags]['2014-11-21 11:19:00':'2014-11-22 06:25:00']
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