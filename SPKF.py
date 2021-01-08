"""
runSPKF: Runs a sigma-point Kalman filter
"""

import numpy as np
import numpy.matlib
from scipy import interpolate
import scipy.io as sio 
from models import DataModel, ModelDyn, spkfData
from funcs import OCVfromSOCtemp, iterSPKF
from pathlib import Path
import matplotlib.pyplot as plt

# read model OCV and model DYN file, previously generated by esctoolbox
modeldyn = ModelDyn.load(Path(f'./modeldyn.pickle'))

# parameters
cellID = 'A123'     # cell identifier
temps = [-25, -15, -5, 5, 15, 25, 35, 45]   # temperatures
mags = [10, 10, 30, 45, 45, 50, 50, 50]     # A123 C-rates e.g 10: 1C, 45: 4.5C
temp = 25

# initialize array to store battery cell data
data = np.zeros(len(mags), dtype=object)

# Load cell-test data to be used for this batch experiment
# Contains variable "DYNData" of which the field "script1" is of 
# interest. This has sub-fields time, current, voltage, soc.
print('Load files')
for idx, itemp in enumerate(temps):
    mag = mags[idx]
    if itemp < 0:
        tempfmt = f'{abs(itemp):02}'
        files = [Path(f'./dyn_data/{cellID}_DYN_{mag}_N{tempfmt}_s1.csv'),
                 Path(f'./dyn_data/{cellID}_DYN_{mag}_N{tempfmt}_s2.csv'),
                 Path(f'./dyn_data/{cellID}_DYN_{mag}_N{tempfmt}_s3.csv')]
        data[idx] = DataModel(itemp, files)
        print(*files, sep='\n')
    else:
        tempfmt = f'{abs(itemp):02}'
        files = [Path(f'./dyn_data/{cellID}_DYN_{mag}_P{tempfmt}_s1.csv'),
                 Path(f'./dyn_data/{cellID}_DYN_{mag}_P{tempfmt}_s2.csv'),
                 Path(f'./dyn_data/{cellID}_DYN_{mag}_P{tempfmt}_s3.csv')]
        data[idx] = DataModel(itemp, files)
        print(*files, sep='\n')

theTemp, = np.where(np.array(temps) == temp)[0]

time = data[theTemp].s1.time
deltat = int(time[1]-time[0])
time = time - time[0] # start time at 0
current = data[theTemp].s1.current # discharge > 0; charge < 0
voltage = data[theTemp].s1.voltage
# Compute soc 
etaParam = modeldyn.etaParam[theTemp]
etaik = data[theTemp].s1.current.copy()
etaik[etaik < 0] = etaParam*etaik[etaik < 0]
Q = data[theTemp].s1.disAh[-1] + data[theTemp].s2.disAh[-1] - data[theTemp].s1.chgAh[-1] - data[theTemp].s2.chgAh[-1]
data[theTemp].Q = Q
soc = 1 - np.cumsum(etaik) * 1/(data[theTemp].Q * 3600)

# Set SOC estimation limits
if False:
  indT = np.where(((soc > 0.05) & (soc < 0.86)))[0]
  soc = soc[indT]
  data[theTemp].soc = soc
  time = time[indT]
  current = current[indT]
  voltage = voltage[indT]

# Reserve storage for computed results, for plotting
sochat = np.zeros((soc.shape))
socbound = np.zeros((soc.shape))

# Covariance values

SigmaX0 = np.diag(np.array([1e-6, 1e-8, 2e-4]))  # uncertainty of initial state
SigmaV = 2e-1 # Uncertainty of voltage sensor, output equation
SigmaW = 2e-1 # Uncertainty of current sensor, state equation

# Create spkfData structure and initialize variables using first
# voltage measurement and first temperature measurement
spkfData = spkfData(voltage[0], temp, SigmaX0, SigmaV, SigmaW, modeldyn)

# Now, enter loop for remainder of time, where we update the SPKF
# once per sample interval
 
for k in range(len(voltage)):
    vk = voltage[k] # "measure" voltage
    ik = current[k] # "measure" current
    Tk = theTemp # "measure" temperature

    # Update SOC (and other model states)
    sochat[k], socbound[k], spkfData = iterSPKF(vk, ik, Tk, deltat, spkfData)

# FIGURE 1
# plot estimate of SOC

if True:
  nanArray = np.empty(time.shape)
  nanArray.fill(np.nan)
  fig = plt.figure(1) # Create plot
  ax = fig.add_subplot(111) # Plot
  ax.plot(time/60, 100*soc, color='black', linewidth=0.01, alpha=0.8)  # Plot (Truth) and customize plot, set linewidth=0.1 for interactive and linewith=0.01 to save figure
  ax.plot(time/60, 100*sochat, color='blue', linewidth=0.01, alpha=0.8)  # Plot (Estimate) and customize plot 
  ax.plot(time/60, 100*(sochat + socbound), color='red', linewidth=0.01, alpha=0.8)  # Plot (Upper-bound) and customize plot
  ax.plot(time/60, 100*(sochat - socbound), color='red', linewidth=0.01, alpha=0.8)  # Plot (lLowe-bound) and customize plot
  plt.grid()
  ax.set(ylabel='SOC (%)', # set labels and axis limits
         xlabel='Time (min)',
        #  xlim=(0, np.ceil(maxtime/60)),
        #  ylim=(10, 80),
        #  xticks = np.arange(0, np.ceil(maxtime/60), 10),
        #  yticks = np.arange(10, 81, 10),
        #  autoscale_on=True
        ) 
  plt.title('SOC estimation using SPKF') #set subplot title    
  plt.tight_layout() 
  plt.savefig('./Figures/SOC estimation using SPKF.pdf', bbox_inches='tight', dpi=1200) # Save plot
  plt.show(block=False) # Show plot 

# Print RMS SOC estimation error 
print('RMS SOC estimation error = ', np.sqrt(np.mean((100*(soc-sochat))**2)), '%')

# FIGURE 2
# Plot estimation error and bounds
if True:
  fig = plt.figure(2)               # Create plot
  bx = fig.add_subplot(111) # Plot
  bx.plot(time/60, 100 * (soc - sochat), color='blue', linewidth=0.01, alpha=0.8)  # Plot (estimation error) and customize plot
  bx.plot(time/60, 100 * socbound, color='red', linewidth=0.01, alpha=0.8)  # Plot (Upper-bound error) and customize plot 
  bx.plot(time/60, 100 * -socbound, color='red', linewidth=0.01, alpha=0.8)  # Plot (Lower-bound error) and customize plot 
  plt.grid()
  bx.set(ylabel='SOC (%)',          # set labels and axis limits
         xlabel='Time (min)',
        #  xlim=(0, np.ceil(maxtime/60)),
        #  ylim=(10, 80),
        #  xticks = np.arange(0, np.ceil(maxtime/60), 10),
        #  yticks = np.arange(10, 81, 10),
        #  autoscale_on=True
        ) 
  plt.title('SOC estimation errors using SPKF') #set subplot title    
  plt.tight_layout() 
  plt.savefig('./Figures/SOC estimation errors using SPKF.pdf', bbox_inches='tight', dpi=1200) # Save plot
  plt.show(block=True) # Show plot 

# Print bounds errors to command window
ind = np.where(abs(soc - sochat) > socbound)[0]
print('Percent of time error outside bounds =', len(ind)/len(soc) * 100, '%')