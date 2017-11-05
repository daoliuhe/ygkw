"""
30/6/2017
Select parameters: k, l, h, tau
result is not stable, use mean value, see transfer2

11/07/2017
use other method rather than kraskov can produce consistent result
"""

from jpype import *
import pandas as pd
import pylab as pl

# location of jar
jarLocation = "C:/Users/40204849/Documents/JIDT/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# read data
data = pd.read_pickle('c145.pkl')
df = data['2014-11-01 00:00:00':'2014-11-22 23:59:00'].resample('10T').mean()
a = df['L0330']  # D133
b = df['P0457']  # F145
c = df['P0451']  # C145
print('Length of data: %s' % len(a))

# parameters
kk = 2
ll = 2
result = []

for tt in range(15):
    hh = tt
    # Create a TE calculator and run it
    teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true")  # Normalise the individual variables
    teCalc.setProperty("k", "4")
    teCalc.setProperty("K_PROP_NAME", "kk")
    teCalc.setProperty("L_PROP_NAME", "ll")
    teCalc.setProperty("K_TAU_PROP_NAME", "tt")
    teCalc.setProperty("L_TAU_PROP_NAME", "tt")
    teCalc.setProperty("DELAY_PROP_NAME", "tt")
    teCalc.initialise()
    # teCalc.setObservations(JArray(JDouble, 1)(a.tolist()), JArray(JDouble, 1)(c.tolist()))
    teCalc.setObservations(a, c)  # numpy array accepted directly, pandas series also accepted
    result.append(teCalc.computeAverageLocalOfObservations())

pl.plot(result, '.-')
pl.title(r'$k=%s, l=%s$' % (kk, ll))
pl.show()
