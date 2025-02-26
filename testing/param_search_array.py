import numpy as np
import harmonic_actions as ha


xmin, xmax = ha.eta_exclusive_bounds(1, 1)

Npts = 30 

#xarr = np.flip(np.linspace(xmin, xmax, Npts))
xarr = np.linspace(xmin, xmax, Npts+2)[1:Npts+1] # Cut off first and last values

np.savetxt('/home/charlie/Documents/ml/CollectiveAction/etavals.txt', xarr)
