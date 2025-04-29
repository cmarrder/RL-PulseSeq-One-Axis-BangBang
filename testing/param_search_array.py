import numpy as np

#import harmonic_actions as ha
#Npts = 30 
#xmin, xmax = ha.eta_exclusive_bounds(1, 1)
#xarr = np.linspace(xmin, xmax, Npts+2)[1:Npts+1] # Cut off first and last values
#xarr = np.flip(np.linspace(xmin, xmax, Npts))
#np.savetxt('/home/charlie/Documents/ml/CollectiveAction/etavals.txt', xarr)

xarr = np.arange(4, 65, 3)
#xarr = np.arange(4, 64, 20)
np.savetxt('/home/charlie/Documents/ml/CollectiveAction/pulse_numbers.txt', xarr, fmt="%i")
