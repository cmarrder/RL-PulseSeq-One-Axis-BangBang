import numpy as np

xmin = 0.01
xmax = 0.4

Npts = 10#100#40

xarr = np.flip(np.linspace(xmin, xmax, Npts))

np.savetxt('tvals.txt', xarr)
