import numpy as np
import matplotlib.pyplot as plt
import os
import pulse_sequence as ps

read_dir = "/home/charlie/Documents/ml/CollectiveAction/data/job_00000" 

freq_file = os.path.join(read_dir, "freq.txt")
filter_file = os.path.join(read_dir, "initialFilter.txt")
state_file = os.path.join(read_dir, "initialState.txt")
time_file = os.path.join(read_dir, "maxTime.txt")
tstep_file = os.path.join(read_dir, "nTimeStep.txt")

approxFreq = np.loadtxt(freq_file)
approxFilter = np.loadtxt(filter_file)
exactTimes = np.loadtxt(state_file)
max_time = np.loadtxt(time_file)
n_time_step = np.loadtxt(tstep_file)

# Do some normalizing:
"""
dt = max_time / n_time_step
globalPhase = np.exp(-1j*2*np.pi*dt/2 * approxFreq) # In our analytical solution, we don't take time to start at 0, but rather at dt/2. 
#approxFtilde *= dt * globalPhase
approxFilter *= globalPhase
"""

# Calculate exact Fourier transform
exactFreq = np.linspace(0, approxFreq[-1], 10*len(approxFreq))
exactFilter = ps.FilterFunc(exactFreq, exactTimes, max_time)
print(len(approxFreq))

plt.plot(approxFreq, approxFilter, label="approx")
plt.plot(exactFreq, exactFilter, label="exact", linestyle="dashed")
plt.xlabel("freq")
plt.ylabel("filter function")
plt.yscale('log')
plt.legend()
plt.show()
