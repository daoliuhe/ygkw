"""
21/3/2017
data is mean-centered only
pca result remains the same. mean-centering does not affect pca.
"""
import pandas as pd
import numpy as np
import pylab as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

path = "C:/Users/40204849/Desktop/YEA Work/03 Time Series Prediction/PV_MV_SV_Full_NUS.csv"
data = pd.read_csv(path, header=5, index_col=0, low_memory=False)
data.index = pd.to_datetime(data.index)
data = data['2013-07-14 00:00:00':'2013-07-16 00:00:00']  # 2 days, 2880

names = ['F0411_PV', 'F0413_PV', 'L0410_PV', 'P0410_PV', 'P0411_PV',
         'T0410_PV', 'T0411_PV', 'T0412_PV', 'T0413_PV', 'T0414_PV',
         'T0415_PV', 'T0416_PV', 'L0310B_PV', 'F1100_PV', 'P1100_PV']
print('Number of variables: %s' % len(names))
df = data[names]['2013-07-15 03:19:00':'2013-07-15 08:19:00'].values
mu = np.mean(df, axis=0)
df = df - mu[np.newaxis, :]

pca = PCA(svd_solver='full')
df1 = pca.fit_transform(df)
colors = ['navy', 'orange']

# after alarm
start = pd.to_datetime('2013-07-15 09:00:00')
end = pd.to_datetime('2013-07-15 10:00:00')
for step in range(12):
    start_ = start + pd.Timedelta('60 min') * step
    end_ = end + pd.Timedelta('60 min') * step
    df_new = data[names][start_:end_].values
    df_new = df_new - mu[np.newaxis, :]
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
