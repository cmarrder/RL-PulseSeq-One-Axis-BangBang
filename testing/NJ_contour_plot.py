import numpy as np
import plot_tools as pt
import pulse_sequence as ps
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.linewidth'] = 4
mpl.rcParams['xtick.major.size'] = 7#5
mpl.rcParams['xtick.major.width'] = 2 
mpl.rcParams['xtick.minor.size'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 7#5
mpl.rcParams['ytick.major.width'] = 2 
mpl.rcParams['ytick.minor.size'] = 1
mpl.rcParams['ytick.minor.width'] = 1
plt.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['font.sans-serif'] = ['Droid Sans']

#dd = '/home/charlie/Documents/ml/CollectiveAction/data_NJ_scan_1overf'
dd = '/home/charlie/Documents/ml/CollectiveAction/data_NJ_scan_lorentzians_run1'
max_time = 1
IR_cutoff = 1 / max_time
S0 = 1
noise_func = lambda x: S0 / np.where(x <= IR_cutoff, 1, x)


#centers = np.array([0.0, 2.0, 6.5, 18.5, 24.0, 30.5, 32.0, 34.5, 35.5, 39.0, 42.0])
#fwhms = np.array([0.97791257, 0.69318029, 1.33978924, 1.22886943, 1.04032235, 0.57028931, 1.85807798, 2.40010327, 0.71804862, 1.9982775, 0.76180626])
#heights = np.array([10.0, 7.07106781, 3.9223227, 2.32495277, 2.04124145, 1.81071492,
#        1.76776695, 1.70251306, 1.67836272, 1.60128154, 1.5430335])
#noise_func = lambda v: ps.lorentzians(v, centers, fwhms, heights=heights)


#save = '../paper_plots/Figure4a.svg'
save = '../paper_plots/Figure4b.svg'
pt.num_pulse_vs_harmonic(dd, noise_func, save=save)
