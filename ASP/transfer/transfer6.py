"""
11/7/2017
Signal de-noising first, and not use kraskov method

"""
from jpype import *
import pandas as pd
import pylab as pl
import numpy as np
from pool import data_filter

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

f_size = 10
af = data_filter(a, filter_size=f_size)
bf = data_filter(b, filter_size=f_size)
cf = data_filter(c, filter_size=f_size)

source = bf
destination = cf

teCalcClass = JPackage("infodynamics.measures.continuous.gaussian").TransferEntropyCalculatorGaussian
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
delay=10
teCalc.setProperty("DELAY", str(delay))

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