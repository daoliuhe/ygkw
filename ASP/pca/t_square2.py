"""
15/8/2017
Kernel PCA
    good:       poly, linear, sigmoid
    not good:   rbf
"""

import pandas as pd
import numpy as np
import pylab as pl
from sklearn.decomposition import PCA, KernelPCA


def c145_data():
    data = pd.read_pickle('../pkl/c145.pkl')
    tags = ['P0451', 'P0452', 'P0452A', 'P0453', 'P0453A',
            'P0455', 'P0465', 'T0413', 'T0450A', 'T0451',
            'T0451A', 'T0452', 'T0452A', 'T0453', 'T0458',
            'T0459', 'T0460', 'X0451A', 'X0451B', 'X0451C',
            'X0451D', 'X0451E', 'X0451F', 'L0330', 'P0457']
    cols = list(set(tags) & set(data.columns))  # some tags not in columns
    return data[cols]


def t141_data():
    data = pd.read_pickle('../pkl/t141.pkl')
    tags = ['L0310B', 'L0311B', 'P0310B', 'T0310B', 'F1100',
            'P1100', 'X1100', 'F0411', 'F0413', 'L0410',
            'P0410', 'P0411', 'T0410', 'T0411', 'T0412',
            'T0413', 'T0414', 'T0415', 'T0416']
    cols = list(set(tags) & set(data.columns))  # some tags not in columns
    return data[cols]


def c145_tq(data, n_comp):
    train = data['2014-11-21 05:19:00':'2014-11-21 11:18:00']  # 6 hrs training data
    test = data['2014-11-21 11:19:00':'2014-11-22 06:25:00']
    t_square_q(train, test, n_comp)


def t141_tq(data, n_comp):
    train = data['2013-07-15 02:20:00':'2013-07-15 08:19:00']
    test = data['2013-07-15 08:20:00':'2013-07-15 23:30:00']
    t_square_q(train, test, n_comp)


def t_square_q(train_data, test_data, n_comp):
    train = train_data.loc[:, train_data.std() >= 0.05]
    means = train.mean(axis=0)
    stds = train.std(axis=0)
    train = (train - means) / stds

    pca = KernelPCA(kernel='sigmoid')
    pca.fit_transform(train)
    lambdas = pca.lambdas_  # all variances

    # T-squared
    test = test_data.loc[:, train_data.std() >= 0.05]
    test = (test - means) / stds

    test_pca = pca.transform(test)
    result = pd.DataFrame(index=test.index)
    result['T2'] = [np.sum(x[:n_comp] ** 2 / lambdas[:n_comp]) for x in test_pca]

    # Q
    non_zeros = (lambdas > 1e-10) * 1
    n = np.sum(non_zeros)  # number of nonzero eigenvalues
    result['Q'] = [np.sum(x[:n] ** 2) - np.sum(x[:n_comp] ** 2) for x in test_pca]

    plot_tq(result)


def plot_tq(res):
    f, axarr = pl.subplots(2, sharex=True, figsize=(14, 10))
    axarr[0].plot(res.T2)
    axarr[0].set_title(res.columns[0])
    axarr[1].plot(res.Q)
    axarr[1].set_title(res.columns[1])
    pl.show()


def explained_variance_ratio(data):
    # 6 hrs data
    df = data['2014-11-21 05:19:00':'2014-11-21 11:18:00']
    # df = data['2013-07-15 02:20:00':'2013-07-15 08:19:00']
    # remove low variance columns, std < 0.05
    df = df.loc[:, df.std() >= 0.05]
    print('Number of training features: %s' % df.shape[1])

    pca = KernelPCA(kernel='poly')
    df_pca = pca.fit_transform(df)
    print('Number of lambdas: %s' % len(pca.lambdas_))
    print('Number of PCs: %s' % df_pca.shape[1])
    comp_ratio = pca.lambdas_ / np.sum(pca.lambdas_)
    cum_ratio = np.cumsum(comp_ratio)  # cumulative explained variance ratio
    n = len(cum_ratio)
    pl.plot(range(1, n + 1), cum_ratio, '-o', linewidth=2.0)
    pl.xlabel('No. of principle components')
    pl.ylabel('Cumulative explained variance ratio')
    pl.show()


if __name__ == '__main__':
    c145 = c145_data()
    c145_tq(c145, 12)
    # t141 = t141_data()
    # t141_tq(t141, 2)

    # explained_variance_ratio(c145)




