"""
12/7/2017
wavelet decomposition
"""

import numpy as np
import pylab as pl
import pandas as pd
from pool import wav_dec

data = pd.read_pickle('../pkl/c145.pkl')
s = data['L0330']['2014-11-15 00:00:00':'2014-11-22 23:59:00'].values

a3, d3, d2, d1 = wav_dec(s, 'haar', 3)

f, axarr = pl.subplots(5, sharex=True, figsize=(12,8))
axarr[0].plot(s)
axarr[0].set_title('Signal')
axarr[1].plot(a3)
axarr[1].set_title('A3')
axarr[2].plot(d3)
axarr[2].set_title('D3')
axarr[3].plot(d2)
axarr[3].set_title('D2')
axarr[4].plot(d1)
axarr[4].set_title('D1')

pl.show()
