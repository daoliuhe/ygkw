"""
30/6/2017
Select parameters: k, l, h, tau
Use mean value to reduce uncertainty

11/07/2017
use other method rather than kraskov can produce consistent result

"""

from jpype import *
import pandas as pd
import pylab as pl
import numpy as np

# location of jar
jarLocation = "C:/Users/40204849/Documents/JIDT/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# read data
data = pd.read_pickle('c145.pkl')
# df = data['2014-11-15 00:00:00':'2014-11-22 23:59:00'].resample('10T').mean()
df = data['2014-09-22 00:00:00':'2014-10-01 23:59:00'].resample('10T').mean()
a = df['L0330']  # D133
b = df['P0457']  # F145
c = df['T0451A']  # C145
print('Length of data: %s' % len(a))

# parameters
kk = 1
ll = 2
tt = 1
result = []

for hh in range(15):
    temp = []
    for i in range(10):
        teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
        teCalc = teCalcClass()
        teCalc.setProperty("NORMALISE", "true")  # Normalise the individual variables
        teCalc.setProperty("k", "4")
        teCalc.setProperty("K_PROP_NAME", "kk")
        teCalc.setProperty("L_PROP_NAME", "ll")
        teCalc.setProperty("K_TAU_PROP_NAME", "tt")
        teCalc.setProperty("L_TAU_PROP_NAME", "tt")
        teCalc.setProperty("DELAY_PROP_NAME", "hh")
        teCalc.initialise()
        teCalc.setObservations(a, c)  # numpy array accepted directly, pandas series also accepted
        temp.append(teCalc.computeAverageLocalOfObservations())

    result.append(np.mean(temp))

pl.plot(result, '.-')
pl.title(r'$k=%s, l=%s, t=%s$' % (kk, ll, tt))
pl.show()
