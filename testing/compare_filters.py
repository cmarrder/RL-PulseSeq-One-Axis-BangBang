import numpy as np
import matplotlib.pyplot as plt
import pulse_sequence as ps
import os

job_dir = '/home/charlie/Documents/ml/CollectiveAction/data'
freqs_arr = np.loadtxt(os.path.join(job_dir, 'freq.txt'))
initial_state = np.loadtxt(os.path.join(job_dir, 'initialState.txt'))
initial_filter = np.loadtxt(os.path.join(job_dir, 'initialFilter.txt'))
max_time = np.loadtxt(os.path.join(job_dir, 'maxTime.txt'))

filter_python = ps.FilterFunc(freqs_arr, initial_state, max_time)

fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(freqs_arr, initial_filter, label='cpp')
ax1.plot(freqs_arr, filter_python, label='python')
ax1.set_xlabel(r'$\nu$')
ax1.set_ylabel(r'$F(\nu)$')
ax1.set_yscale('log')
ax1.legend()

ax2.plot(freqs_arr, np.abs(initial_filter - filter_python))
ax2.set_label(r'$\nu$')
ax2.set_ylabel(r'$|F(\nu)_{python} - F(\nu)_{C++}|$')

plt.show()
