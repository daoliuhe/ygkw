"""
4/4/2017
data source: ValidAlarm.csv
Find the most occurrences of tags
Use column tag as index is easy
"""
import pandas as pd
from collections import Counter
import pylab as pl

path = "C:/Users/40204849/Desktop/Time Series Prediction/new data/ValidAlarm.csv"
data = pd.read_csv(path, index_col=1)
counter = Counter(data.index).most_common(5)
tags = [i[0] for i in counter]
nums = [i[1] for i in counter]
pl.figure(figsize=(12, 8))
pl.stem(nums)
pl.xlim([-1, 5])
for i in range(len(tags)):
    pl.text(i+0.1, nums[i], str(nums[i]))
pl.xticks(range(6), tags)
pl.show()

