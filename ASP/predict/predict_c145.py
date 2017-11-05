"""
27/4/2017
data period, including c145 and t141
    a. 2013-07-13 00:00:00 - 2013-07-18 00:00:00
    b. 2014-11-20 00:00:00 - 2014-11-23 00:00:00

C145 case

Predict every tag of C145
"""

import numpy as np
import pandas as pd


def linear_fit(x, y):
    """ linear regression: y = f(x) """
    w = np.sum((x - np.mean(x)) * (y - np.mean(y))) \
        / np.sum((x - np.mean(x)) * (x - np.mean(x)))
    b = np.mean(y) - w * np.mean(x)
    return w, b

data = pd.read_pickle('../pkl/new_dataC145.pkl')
c145 = data['2014-11-20 00:00:00':'2014-11-23 00:00:00'].copy()

path = "../csv/C145 Limits.csv"
limits = pd.read_csv(path, header=0, index_col=0)

names = limits.index
model_start = pd.to_datetime('2014-11-20 02:00:00')
rng = pd.date_range(start=model_start, end=c145.index[-1], freq='15T')
result = pd.DataFrame(index=rng, columns=names)
for name in names:
    ll = limits.ix[name]['LL']
    hh = limits.ix[name]['HH']
    max_step = len(rng)

    for step in range(max_step):
        end = model_start + pd.Timedelta('15 min') * step
        start = end - pd.Timedelta('2 hours')  # use 2 hours data for training
        y = c145[name][start:end]
        x = np.linspace(1, len(y), len(y))
        w, b = linear_fit(x, y)

        if y[0] >= y[-1]:  # downward trend
            threshold = ll  # use low low as threshold
            if np.min(y) < threshold:  # y crossed threshold
                time_pred = 'Not valid'
            else:
                if w >= 0:  # it's possible but not good, no intersection with ll
                    time_pred = 'More than 2 days'
                else:
                    t = (threshold - b) / w
                    if t > 3000:  # over 2 days
                        time_pred = 'More than 2 days'
                    else:
                        time_pred = start + pd.Timedelta(str(t.astype(int)) + 'min')

        else:  # upward trend
            threshold = hh  # use high high as threshold
            if np.max(y) > threshold:  # y crossed threshold
                time_pred = 'Not valid'
            else:
                if w <= 0:
                    time_pred = 'More than 2 days'  # no intersection with hh
                else:
                    t = (threshold - b) / w
                    if t > 3000:  # over 2 days
                        time_pred = 'More than 2 days'
                    else:
                        time_pred = start + pd.Timedelta(str(t.astype(int)) + 'min')

        result.ix[end, name] = time_pred

result.to_csv('../csv/c145_result.csv')


