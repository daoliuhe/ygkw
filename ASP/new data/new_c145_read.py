"""
4/4/2017
New data: Jan, Feb 2017
read new data

"""
import pandas as pd
import numpy as np

path = "C:/Users/40204849/Desktop/Time Series Prediction/new data/List 2 C145.csv"
data = pd.read_csv(path, header=6, index_col=0)
data.index = pd.to_datetime(data.index, dayfirst=True)

# extract tag names
names = [col[-9:-3] for col in data.columns]
tags = []
for name in names:
    if name[0] == '.':
        tags.append(name[1:].upper())
    else:
        tags.append(name.upper())

print(tags)

df = pd.DataFrame(data=data.values, index=data.index, columns=tags, dtype=np.float32)
df = df.drop('NAMED:', 1)
df.to_pickle('new_c145.pkl')
