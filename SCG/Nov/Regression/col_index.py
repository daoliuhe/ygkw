"""

Total: 66
Remove repeated: PDI382-05.PV, TI392-03A.PV
Now: 66 - 2 = 64

3/11/2016
one index <---> one variable

8/12/2016
col_index_new: remove vibration signals from col_index
Now: 64 - 2 = 62

"""

import pickle

# headers is redundant, including repeated variables
# headers is dict, key: column index, value: variable name
headers = pickle.load(open("../../Data/headers.p", "rb"))
# shortlist has removed repeated variables
shortlist = [line.rstrip('\n') for line in
             open('../../Data/headers.txt')]  # text file without repeated names

col_index = []
for name in shortlist:
    # return col no. from name
    col = list(headers.keys())[list(headers.values()).index(name)]
    col_index.append(col)

col_index_new = []
for index in col_index:
    if index == 41:  # XI392-04A
        pass
    elif index == 42:  # XI392-04B
        pass
    else:
        col_index_new.append(index)

pickle.dump(col_index, open("col_index.p", "wb"))
pickle.dump(col_index_new, open("col_index_new.p", "wb"))
