"""
26/7/2017
one class svm
pca first, then svm
"""

import pandas as pd
import numpy as np
import pylab as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


data = pd.read_pickle('../pkl/c145.pkl')
names = ['P0451', 'P0452', 'P0452A', 'P0453', 'P0453A',
         'P0455', 'P0465', 'T0413', 'T0450A', 'T0451',
         'T0451A', 'T0452', 'T0452A', 'T0453', 'T0458',
         'T0459', 'T0460']
df = data[names].copy()
df -= df.mean(axis=0)

# train - data before alarm, test - 1 hr
x_train = df['2014-11-20 11:19:00':'2014-11-21 11:18:00'].values
x_test = df['2014-11-21 17:00:00':'2014-11-21 18:00:00'].values

pca = PCA()
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
colors = ['navy', 'orange', 'green']

n_comp = 2
clf = OneClassSVM(random_state=42)
clf.fit(x_train[:, :n_comp])
train_pred = clf.predict(x_train[:, :n_comp])
test_pred = clf.predict(x_test[:, :n_comp])

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-4, 4, 100))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

pl.figure(figsize=(12, 8))
pl.title("OneClassSVM")
pl.contourf(xx, yy, Z, cmap=pl.cm.Blues_r)
a = pl.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')

b1 = pl.scatter(x_train[:, 0], x_train[:, 1], c='white',
                s=40, edgecolor='k')
b2 = pl.scatter(x_test[:, 0], x_test[:, 1], c='red',
                s=40, edgecolor='k')
pl.axis('tight')
pl.xlim((-10, 10))
pl.ylim((-4, 4))
pl.legend([b1, b2],
          ["training observations",
           "testing observations"],
          loc="upper left")
pl.show()