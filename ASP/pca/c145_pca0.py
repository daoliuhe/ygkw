"""
16/3/2017
explained variance ratio vs no. of components

"""
import pandas as pd
import numpy as np
import pylab as pl
from sklearn.decomposition import PCA

path = "C:/Users/40204849/Desktop/YEA WORK/03 Time Series Prediction/PVC145Case.csv"
data = pd.read_csv(path, header=0, index_col=0)
data.index = pd.to_datetime(data.index)

names = ['P0453', 'P0453A', 'P0455', 'T0413',
         'T0450A', 'T0451', 'T0451A', 'T0452',
         'T0452A', 'T0453', 'Flow']
print('Number of variables: %s' % len(names))

# before alarm
df = data[names]['2014-11-21 08:19:00':'2014-11-21 11:18:00'].values
pca = PCA()
pca.fit(df)

comp_ratio = pca.explained_variance_ratio_
cum_ratio = np.cumsum(comp_ratio)  # cumulative explained variance ratio
pl.plot(range(df.shape[1]), cum_ratio, linewidth=2.0)
pl.xlabel('No. of components')
pl.ylabel('Cumulative explained variance ratio')
pl.show()

