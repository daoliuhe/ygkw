"""

one datetime <--> one row number

27/10/2016: Return row number given datetime

"""

# import pickle

data = [line.rstrip('\n') for line in open('./Old/datetime.txt')]
datetimes = {}
for row in range(0, len(data)):
    datetimes[data[row]] = row

print(datetimes['14/2/2016 12:01 AM'])
print(datetimes['25/3/2016 12:00 AM'])

# pickle.dump(datetimes, open("datetimes.p", "wb"))
