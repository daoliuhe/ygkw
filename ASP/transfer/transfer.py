"""
28/6/2017
Calculate transfer entropy between signals

1. normalise or not, doesn't matter, result is close
2. numpy array objects accepted, pandas Series also accepted, no need to convert
3. data length changes TE value, but not very much

"""
from jpype import *
import pandas as pd

# location of jar
jarLocation = "C:/Users/40204849/Documents/JIDT/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# read data
data = pd.read_pickle('c145.pkl')
df = data['2014-11-10 00:00:00':'2014-11-22 23:59:00'].resample('10T').mean()
a = df['L0330']  # D133
b = df['P0457']  # F145
c = df['P0451']  # C145
print('Length of data: %s' % len(a))

# Create a TE calculator and run it
teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
teCalc.setProperty("k", "4") # Use Kraskov parameter K=4 for 4 nearest points
teCalc.setProperty("K_PROP_NAME", "2")
teCalc.initialise() # Use history length 1 (Schreiber k=1)
# teCalc.setObservations(JArray(JDouble, 1)(a.tolist()), JArray(JDouble, 1)(c.tolist()))
teCalc.setObservations(a, c)  # numpy array accepted directly, pandas series also accepted
result = teCalc.computeAverageLocalOfObservations()
print("TE result: %.4f nats" % result)



