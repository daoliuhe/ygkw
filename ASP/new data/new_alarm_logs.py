"""
4/4/2017
data source: ValidAlarm.csv
read alarm logs
"""
import pandas as pd

path = "C:/Users/40204849/Desktop/Time Series Prediction/new data/ValidAlarm.csv"
data = pd.read_csv(path, index_col=0)

ind = [i[:19] for i in data.index]  # remove t zone tail
df = pd.DataFrame(data=data.values, index=ind, columns=data.columns)
df.index = pd.to_datetime(df.index)