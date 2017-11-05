"""
30/6/2017
who causes who???

3/7/2017
result shows that C145 causes F145 and D133, F145 causes D133.
It is just the opposite to the fact.
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
df = data['2014-11-01 00:00:00':'2014-11-22 23:59:00'].resample('10T').mean()
a = df['L0330']  # D133
b = df['P0457']  # F145
c = df['T0451A']  # C145
print('Length of data: %s' % len(a))

source = a
destination = c

teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true")  # Normalise the individual variables
teCalc.setProperty("k", "4")
teCalc.setProperty("k_HISTORY", "2")
teCalc.setProperty("l_HISTORY", "2")
teCalc.setProperty("DELAY", "4")

teCalc.initialise()
teCalc.setObservations(source, destination)  # numpy array accepted directly, pandas series also accepted
te1 = teCalc.computeAverageLocalOfObservations()

teCalc.initialise()
teCalc.setObservations(destination, source)  # numpy array accepted directly, pandas series also accepted
te2 = teCalc.computeAverageLocalOfObservations()

print('Source to destination: %.4f' % te1)
print('Destination to source: %.4f' % te2)

diff = te1 - te2
if diff >= 0:
    print('Causality measure is %.4f: source causes destination, reasonable' % diff)
else:
    print('Causality measure is %.4f: destination causes source, weird' % diff)

