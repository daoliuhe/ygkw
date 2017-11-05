"""
4/7/2017
T141 case study
1. D131B causes T141, confirmed by transfer entropy
    Length of data: 2160
    Source to destination: 0.0297
    Destination to source: -0.0115
    Causality measure is 0.0412: source causes destination, reasonable
2. D131B causes R111, confirmed by transfer entropy
3. R111 causes T141, confirmed by transfer entropy
4. transfer entropy values are too small

"""

from jpype import *
import pandas as pd
import pylab as pl
import numpy as np

# location of jar
jarLocation = "C:/Users/40204849/Documents/JIDT/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# T141 case
data = pd.read_pickle('t141.pkl')
tags = ['L0310B', 'L0311B', 'P0310B', 'T0310B',
        'F1100', 'P1100',
        'L0410', 'T0410', 'P0411', 'F0411', 'F0413', 'T0414', 'T0411']
df_ = data[tags].astype(float)
df = df_['2013-07-01 00:00:00':'2013-07-15 23:59:00'].resample('10T').mean()
a = df['L0310B']  # D131B
b = df['F1100']  # R111
c = df['L0410']  # T141
print('Length of data: %s' % len(a))

source = a
destination = c

teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true")  # Normalise the individual variables
teCalc.setProperty("k", "4")
teCalc.setProperty("k_HISTORY", "2")
teCalc.setProperty("l_HISTORY", "2")
teCalc.setProperty("DELAY", "1")

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

