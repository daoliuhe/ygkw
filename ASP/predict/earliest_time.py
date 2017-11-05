"""
3/8/2017
After alarm occurs, for each step, find the earliest prediction time
"""

import pandas as pd


def c145_time():
    data = pd.read_csv('../csv/c145_result.csv', index_col=0, parse_dates=True)

    start = pd.to_datetime('2014-11-21 11:30:00')
    end = pd.to_datetime('2014-11-22 06:30:00')
    c145 = data[data.columns[:-2]][start:end]   # choose C145 tags

    pred_time = pd.DataFrame(index=c145.index)
    predTime = [min(c145.values[i]) for i in range(len(c145))]
    pred_time['C145'] = predTime
    pred_time['D133'] = data['L0330'][start:end]
    pred_time['F145'] = data['P0457'][start:end]
    pred_time.to_csv('../csv/c145_pred_time.csv')


def t141_time():
    data = pd.read_csv('../csv/t141_result.csv', index_col=0, parse_dates=True)

    start = pd.to_datetime('2013-07-14 01:45:00')
    end = pd.to_datetime('2013-07-15 23:30:00')
    t141 = data[data.columns[:-8]][start:end]       # t141 tags
    d131b = data[data.columns[-8:-4]][start:end]    # d131b tags
    r111 = data[data.columns[-4:-1]][start:end]     # r111 tags

    pred_time = pd.DataFrame(index=t141.index)
    pred_time['T141'] = [min(t141.values[i]) for i in range(len(t141))]
    pred_time['D131B'] = [min(d131b.values[i]) for i in range(len(d131b))]
    pred_time['R111'] = [min(r111.values[i]) for i in range(len(r111))]
    pred_time.to_csv('../csv/t141_pred_time.csv')


if __name__ == '__main__':
    c145_time()
    t141_time()


