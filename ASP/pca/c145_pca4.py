"""
27/03/2017
Kernel PCA
Not better than basic PCA in this case
"""

import pandas as pd
import numpy as np
import pylab as pl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

path = "C:/Users/40204849/Desktop/Time Series Prediction/PVC145Case.csv"
data = pd.read_csv(path, header=0, index_col=0)
data.index = pd.to_datetime(data.index)

names = ['P0453', 'P0453A', 'P0455', 'T0413',
         'T0450A', 'T0451', 'T0451A', 'T0452',
         'T0452A', 'T0453', 'Flow']
print('Number of variables: %s' % len(names))

# before alarm
df = data[names]['2014-11-21 08:19:00':'2014-11-21 11:18:00'].values
pca = KernelPCA(n_components=4, kernel='cosine')
df1 = pca.fit_transform(df)
colors = ['navy', 'orange']

# after alarm
start = pd.to_datetime('2014-11-21 12:00:00')
end = pd.to_datetime('2014-11-21 13:00:00')
for step in range(10):
    start_ = start + pd.Timedelta('60 min') * step
    end_ = end + pd.Timedelta('60 min') * step
    df_new = data[names][start_:end_].values
    df2 = pca.transform(df_new)
    pl.figure(step+1, figsize=(12, 8))
    pl.scatter(df1[:, 0], df1[:, 1], color=colors[0])
    pl.scatter(df2[:, 0], df2[:, 1], color=colors[1])
    pl.title(str(start_)[11:19] + '-' + str(end_)[11:19])
    pl.xlabel('First principle component')
    pl.ylabel('Second principle component')
    # fig_name = str(step + 12)
    # pl.savefig(fig_name + '.jpg', dpi=300)

pl.show()



