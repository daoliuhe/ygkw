"""
29/8/2017
Calculate Q contributions
"""

import pandas as pd
import numpy as np
import pylab as pl


def c145_data():
    data = pd.read_pickle('../pkl/c145.pkl')
    tags = ['P0451', 'P0452', 'P0452A', 'P0453', 'P0453A',
            'P0455', 'P0465', 'T0413', 'T0450A', 'T0451',
            'T0451A', 'T0452', 'T0452A', 'T0453', 'T0458',
            'T0459', 'T0460', 'X0451A', 'X0451B', 'X0451C',
            'X0451D', 'X0451E', 'X0451F']
    cols = list(set(tags) & set(data.columns))  # some tags not in columns
    return data[cols]


def t141_data():
    data = pd.read_pickle('../pkl/t141.pkl')
    tags = ['F0411', 'F0413', 'L0410',
            'P0410', 'P0411', 'T0410', 'T0411', 'T0412',
            'T0413', 'T0414', 'T0415', 'T0416']
    cols = list(set(tags) & set(data.columns))  # some tags not in columns
    return data[cols]


def c145_q_con(data, n_comp):
    train = data['2014-11-21 05:19:00':'2014-11-21 11:18:00']  # 6 hrs training data
    test = data['2014-11-21 11:19:00':'2014-11-22 06:25:00']
    q_contributions(train, test, n_comp)


def t141_q_con(data, n_comp):
    train = data['2013-07-15 02:20:00':'2013-07-15 08:19:00']
    test = data['2013-07-15 08:20:00':'2013-07-15 23:30:00']
    q_contributions(train, test, n_comp)


def q_contributions(train_data, test_data, n_comp):
    # Q contributions show how much each variable contributes to the overall Q statistic

    train = train_data.loc[:, train_data.std() >= 0.05]
    means = train.mean(axis=0)
    stds = train.std(axis=0)
    train = (train - means) / stds  # data standardization

    test = test_data.loc[:, train_data.std() >= 0.05]
    test = (test - means) / stds
    tags = train.columns

    cov = np.dot(train.T, train) / train.shape[0]
    u, s, v = np.linalg.svd(cov.astype(float))
    p = u[:, :n_comp]
    pp = np.dot(p, p.T)

    err = test.values - np.dot(test.values, pp)
    err2 = err ** 2
    q_contri = np.array([e / np.sum(e) for e in err2])
    q_sort_index = np.argsort(-q_contri, axis=1)
    q_sort = [list(tags[index]) for index in q_sort_index]
    result = pd.DataFrame(data=q_sort, index=test.index, columns=range(len(tags)))
    # result.to_csv('c145_q_con.csv')
    result.to_csv('t141_q_con.csv')


if __name__ == '__main__':
    # c145 = c145_data()
    # c145_q_con(c145, 12)
    t141 = t141_data()
    t141_q_con(t141, 2)

