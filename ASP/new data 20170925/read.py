"""
25/9/2017
New data arrived at Sep 25, with more failure cases.
Save data to pkl folder
    c145_trip1.pkl
    c145_trip2.pkl
    c145_trip3.pkl

"""

import pandas as pd
import numpy as np


def read_data(path):
    data = pd.read_csv(path, header=6, index_col=0, low_memory=False)
    data = data[~ data.index.isnull()]  # remove blank lines
    data.index = pd.to_datetime(data.index, dayfirst=True)

    # extract tag names
    cols = [col for col in data.columns if col[-2:] == 'PV']  # remove invalid columns
    names = [col[-9:-3] for col in cols]
    tags = [name[1:].upper() if name[0] == '.' else name.upper()
            for name in names]

    df = pd.DataFrame(data=data[cols].values, index=data.index,
                      columns=tags, dtype=np.float32)
    df = df.drop(['D0481A', 'D0481B', 'D0481C', 'Q1024', 'Q2024'], 1)  # del columns

    return df


if __name__ == '__main__':
    path = '../csv/C145 Trip PV 3 of 3.csv'
    df = read_data(path)
    df.to_pickle('../pkl/c145_trip3.pkl')


