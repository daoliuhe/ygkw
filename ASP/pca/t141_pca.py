"""
17/3/2017
pca explained ratio

21/3/2017
standardize reduce explained ratio a lot
"""
import pandas as pd
import numpy as np
import pylab as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

path = "C:/Users/40204849/Desktop/Time Series Prediction/PV_MV_SV_Full_NUS.csv"
data = pd.read_csv(path, header=5, index_col=0, low_memory=False)
data.index = pd.to_datetime(data.index)
data = data['2013-07-14 00:00:00':'2013-07-16 00:00:00']  # 2 days, 2880

names = ['F0411_PV', 'F0413_PV', 'L0410_PV', 'P0410_PV', 'P0411_PV',
         'T0410_PV', 'T0411_PV', 'T0412_PV', 'T0413_PV', 'T0414_PV',
         'T0415_PV', 'T0416_PV', 'L0310B_PV', 'F1100_PV', 'P1100_PV']
print('Number of variables: %s' % len(names))

df = data[names]['2013-07-15 03:19:00':'2013-07-15 08:19:00']
# s = StandardScaler()
# df = s.fit_transform(df)
pca = PCA()
pca.fit_transform(df)

comp_ratio = pca.explained_variance_ratio_
cum_ratio = np.cumsum(comp_ratio)  # cumulative explained variance ratio
pl.plot(range(1, df.shape[1]+1), cum_ratio, linewidth=2.0)
pl.xlabel('No. of components')
pl.ylabel('Cumulative explained variance ratio')
pl.show()