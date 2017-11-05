"""
16/3/2017
plot pca results using animation
"""

import pandas as pd
import numpy as np
import pylab as pl
from sklearn.decomposition import PCA
import matplotlib.animation as animation

path = "C:/Users/40204849/Desktop/YEA Work/03 Time Series Prediction/PVC145Case.csv"
data = pd.read_csv(path, header=0, index_col=0)
data.index = pd.to_datetime(data.index)

names = ['P0453', 'P0453A', 'P0455', 'T0413',
         'T0450A', 'T0451', 'T0451A', 'T0452',
         'T0452A', 'T0453', 'Flow']
print('Number of variables: %s' % len(names))

# before alarm
df = data[names]['2014-11-21 08:19:00':'2014-11-21 11:18:00'].values
pca = PCA(n_components=4, svd_solver='full')
df1 = pca.fit_transform(df)
fig1 = pl.figure(figsize=(12, 8))
pl.scatter(df1[:, 0], df1[:, 1])

# after alarm
df_new = data[names]['2014-11-21 15:30:00':'2014-11-21 19:00:00'].values
df2 = pca.transform(df_new)


def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

data_plot = df2[:, :2].T
l, = pl.plot([], [], 'r-o')
pl.xlim(-4, 14)
pl.ylim(-14, 4)
line_ani = animation.FuncAnimation(fig1, update_line, len(df2), fargs=(data_plot, l),
                                   interval=200, blit=True, repeat=False)
pl.title('15:30:00 - 19:00:00')

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
line_ani.save('pca.mp4', writer=writer)
pl.show()
