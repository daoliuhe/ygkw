"""
3/7/2017
Do series differencing first, then calculate transfer entropy

1. Causality measure is -0.0133: destination causes source, weird
2. result is not consistent, sometime positive sometime negative, Nov 15-22
3. result is not consistent, sometime positive sometime negative, Nov 1-22
4. overall, result is not consistent, differencing is NOT working

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
df = data['2014-11-01 00:00:00':'2014-11-22 23:59:00'].resample('30T').mean()
a = df['L0330']  # D133
b = df['P0457']  # F145
c = df['T0451A']  # C145
print('Length of data: %s' % len(a))

# differencing
a_diff = a.diff()[1:]
b_diff = b.diff()[1:]
c_diff = c.diff()[1:]

source = a_diff
destination = c_diff


teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables

teCalc.initialise() # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
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