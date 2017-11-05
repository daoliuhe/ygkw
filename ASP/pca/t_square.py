"""
8/8/2017
StandardScaler {AS SAME AS} data -= data.mean(axis=0), data /= data.std(axis=0)

10/8/2017
remove columns with low variance

16/8/2017
pca.explained_variances_ are exactly eigenvalues
if cov is normalized by n, not n - 1
note that data should be standardized first

Find another way to calculate T2, very good, just use formula

18/8/2017
Speaking of T2 limit, chi square and F distribution have close values

"""

import pandas as pd
import numpy as np
import pylab as pl
from sklearn.decomposition import PCA


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


def explained_variance_ratio(data):
    # 6 hrs data
    df = data['2014-11-21 05:19:00':'2014-11-21 11:18:00']
    # df = data['2013-07-15 02:20:00':'2013-07-15 08:19:00']
    # remove low variance columns, std < 0.05
    df = df.loc[:, df.std() >= 0.05]

    pca = PCA()
    pca.fit_transform(df)
    comp_ratio = pca.explained_variance_ratio_
    cum_ratio = np.cumsum(comp_ratio)  # cumulative explained variance ratio
    n = df.shape[1]
    pl.plot(range(1, n + 1), cum_ratio, '-o', linewidth=2.0)
    pl.xlabel('No. of principle components')
    pl.ylabel('Cumulative explained variance ratio')
    pl.show()


def c145_tq(data, n_comp):
    train = data['2014-11-21 05:19:00':'2014-11-21 11:18:00']  # 6 hrs training data
    test = data['2014-11-21 11:19:00':'2014-11-22 06:25:00']
    t_square_q(train, test, n_comp)


def t141_tq(data, n_comp):
    train = data['2013-07-15 02:20:00':'2013-07-15 08:19:00']
    test = data['2013-07-15 08:20:00':'2013-07-15 23:30:00']
    t_square_q(train, test, n_comp)


def plot_tq(res):
    f, axarr = pl.subplots(2, sharex=True, figsize=(14, 10))
    axarr[0].plot(res.T2)
    axarr[0].set_title(res.columns[0])
    axarr[1].plot(res.Q)
    axarr[1].set_title(res.columns[1])
    pl.show()


def t_square_q(train_data, test_data, n_comp):
    train = train_data.loc[:, train_data.std() >= 0.05]
    means = train.mean(axis=0)
    stds = train.std(axis=0)
    train = (train - means) / stds  # data standardization

    pca = PCA(n_components=n_comp)
    pca.fit_transform(train)
    variances = pca.explained_variance_

    # T-squared
    test = test_data.loc[:, train_data.std() >= 0.05]
    test = (test - means) / stds

    test_pca = pca.transform(test)
    result = pd.DataFrame(index=test.index)
    result['T2'] = [np.sum(x ** 2 / variances) for x in test_pca]

    # Q
    cov = np.dot(train.T, train) / train.shape[0]
    u, s, v = np.linalg.svd(cov.astype(float))
    p = u[:, :n_comp]
    pp = np.dot(p, p.T)
    i = np.identity(pp.shape[0])
    result['Q'] = [np.dot(np.dot(x, i - pp), x.T) for x in test.values]

    """
    # alternative to calculate T2, based on formula
    inv = np.linalg.inv(np.diag(s[:n_comp]))
    ppp = np.dot(np.dot(p, inv), p.T)
    result['t2'] = [np.dot(np.dot(x, ppp), x.T) for x in test.values]
    """

    # plot_tq(result)
    t2_statistic(train, n_comp, result, s)


def t2_statistic(train, n_comp, result, lambdas):
    n = len(train)
    # T2 threshold
    f_alpha = np.array([1.7522, 2.185])
    t2_alpha = n_comp * (n - 1) * f_alpha / (n - n_comp)

    # t2_F = n_comp * (n - 1) * 2.9957 / (n - n_comp)
    # t2_chi_sq = 5.99

    # Q threshold
    theta1 = np.sum(lambdas[n_comp:])
    theta2 = np.sum(lambdas[n_comp:] ** 2)
    theta3 = np.sum(lambdas[n_comp:] ** 3)
    h0 = 1 - 2 * theta1 * theta3 / (3 * theta2 * theta2)
    c_alpha = np.array([1.65, 2.33])
    q1 = h0 * c_alpha * np.sqrt(2 * theta2) / theta1
    q2 = theta2 * h0 * (h0 - 1) / (theta1 * theta1)
    q_alpha = theta1 * (q1 + 1 + q2) ** (1 / h0)

    f, axarr = pl.subplots(2, sharex=True, figsize=(14, 10))
    axarr[0].plot(result.T2)
    axarr[0].axhline(y=t2_alpha[0], color='r', linestyle='-')
    axarr[0].axhline(y=t2_alpha[1], color='y', linestyle='-')
    axarr[0].legend(['T2', 'alpha=0.05', 'alpha=0.01'], loc='upper left')

    axarr[1].plot(result.Q)
    axarr[1].axhline(y=q_alpha[0], color='r', linestyle='-')
    axarr[1].axhline(y=q_alpha[1], color='y', linestyle='-')
    axarr[1].legend(['Q', 'alpha=0.05', 'alpha=0.01'], loc='upper left')
    pl.show()


if __name__ == '__main__':
    # c145 = c145_data()
    # c145_tq(c145, 12)
    t141 = t141_data()
    t141_tq(t141, 2)

    # explained_variance_ratio(c145)



