"""
25/9/2017
Plot data trend

"""

import pandas as pd
import pylab as pl


def subplot(m, columns):
    f, axarr = pl.subplots(m, sharex=True, figsize=(12, 8))
    for j in range(m):
        axarr[j].plot(data[columns[j]])
        axarr[j].set_title(columns[j])

data = pd.read_pickle('../pkl/c145.pkl')
n = data.shape[1] // 5
r = data.shape[1] % 5

# for i in range(n):
#     names = data.columns[5*i:5+5*i]
#     subplot(5, names)
#
# if r:
#     rest = data.columns[5*n:]
#     subplot(r, rest)
#
names = ['P0457', 'P0465', 'T0450']
subplot(3, names)

pl.show()