"""
11/10/2017

"""

import pandas as pd


def remove():
    path = 'C:/Users/40204849/Desktop/Current implementation.xlsx'
    data = pd.read_excel(path)
    left = data['LHS_Node1'].tolist()
    right = data['RHS_Node2'].tolist()
    combine = sorted(list(set(zip(left, right))))
    for a, b in combine:
        if (b, a) in combine:
            combine.remove((b, a))
    new_left = [combine[i][0] for i in range(len(combine))]
    new_right = [combine[i][1] for i in range(len(combine))]
    df = pd.DataFrame()
    df['left'] = new_left
    df['right'] = new_right
    df.to_excel('C:/Users/40204849/Desktop/reduced mpr.xlsx',
                index=False, header=False)
