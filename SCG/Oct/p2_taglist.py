"""

Total in tag list: 137
Remove: PDIC389-01.PV, PDIC387-01.PV, ZI392-04A.PV, ZI392-04B.PV, FI389-01.PV
        PI368.PV, PI369.PV, TI370.PV, TI372.PV, XEV394.PV
Now in total: 137-10 = 127

24/10/2016: Return col no. from variable name

"""

import pickle

headers = pickle.load(open("headers.p", "rb"))  # headers are redundant, it's okay
taglist = [line.rstrip('\n') for line in
           open('./Old/variables.txt')]  # text file has been updated

col_index = []
for name in taglist:
    # return col no. from name
    col = list(headers.keys())[list(headers.values()).index(name)]
    col_index.append(col)

pickle.dump(col_index, open("col_index.p", "wb"))
