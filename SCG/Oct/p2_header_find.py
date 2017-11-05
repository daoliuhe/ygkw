"""

Find variable name from column number

Example:
    header = 'TI307B.PV'
    print(list(headers.keys())[list(headers.values()).index(header)])

"""

import pickle
import numpy as np

headers = pickle.load(open("headers.p", "rb"))
cols = pickle.load(open("selected_cols.p", "rb"))

# for i in cols[:30]:
#     print(headers[i])

# print(headers[149])  # vibration signal

header1 = 'XI392-04A.PV'
header2 = 'XI392-04B.PV'
header3 = 'XI392-12.PV'

print(list(headers.keys())[list(headers.values()).index(header1)])
print(list(headers.keys())[list(headers.values()).index(header2)])
print(list(headers.keys())[list(headers.values()).index(header3)])


