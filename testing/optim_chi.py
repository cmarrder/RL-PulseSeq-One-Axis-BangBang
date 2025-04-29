import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pulse_sequence as ps
import plot_tools as pt
import scipy.special as scsp
import scipy.optimize as opt
#import random_pulse_times as rpt

import os

def check_seq_symmetric(pulseTimes, maxTime, symmetryEps):
    # Check if pulse is symmetric
    nPulse = len(pulseTimes)
    symmetry_flag = True
    if (nPulse % 2 == 0):
        for i in range(int(nPulse/2)):
            if abs(pulseTimes[nPulse - 1 - i] - (maxTime - pulseTimes[i])) > symmetryEps:
                print('\nPulse sequence NOT SYMMETRIC:')
                print('    Max time - time of pulse index {}:'.format(i))
                print('    ', maxTime - pulseTimes[i])
                print('    Time of pulse index N - 1 - {}:'.format(i))
                print('    ', pulseTimes[nPulse - 1 - i])
                symmetry_flag = False
        if symmetry_flag == True:
            print('\nPulse sequence SYMMETRIC')
    elif (nPulse % 2 == 1):
        for i in range(int( (nPulse - 1)/2 )):
            if abs(pulseTimes[nPulse - 1 - i] - (maxTime - pulseTimes[i])) > symmetryEps:
                print('\nPulse sequence NOT SYMMETRIC:')
                print('    Max time - time of pulse index {}:'.format(i))
                print('    ', maxTime - pulseTimes[i])
                print('    Time of pulse index N - 1 - {}:'.format(i))
                print('    ', pulseTimes[nPulse - 1 - i])
                symmetry_flag = False
        if abs(pulseTimes[int( (nPulse - 1)/2 )] - maxTime/2) > symmetryEps:
            print('\nPulse sequence NOT SYMMETRIC:')
            print('    Max time / 2')
            print('    ', maxTime/2)
            print('    Time of pulse index (N - 1)/2 ')
            print('    ', pulseTimes[int( (nPulse - 1)/2 )])
            symmetry_flag = False
        if symmetry_flag == True:
            print('\nPulse sequence SYMMETRIC')
    return


def symmetricPulseTimings(timePulsesHalf, maxTime, nPulse):
    """ We want to generate symmetric pulse sequences using the timings from timePulsesHalf.
    For convenience, timePulsesHalf must have even number of elements for both nPulse even and odd.
    This is because if nPulse is odd, then the timing of the middle pulse
    must be maxTime/2 in order for the sequence to be symmetric. Thus, in this odd case,
    we only care about the pulse timings before the middle pulse.
    """
    fullTimes = np.zeros(nPulse)
    fullTimes[0:len(timePulsesHalf)] = timePulsesHalf
    if (nPulse % 2 == 0):
        if 2*len(timePulsesHalf) != nPulse:
            raise ValueError('For symmetric sequences of even nPulse, len(timePulsesHalf) must be nPulse / 2.')
        fullTimes[len(timePulsesHalf):nPulse] = maxTime - np.flip(timePulsesHalf)
    else:
        if 2*len(timePulsesHalf) + 1 != nPulse:
            raise ValueError('For symmetric sequences of odd nPulse, len(timePulsesHalf) must be (nPulse - 1) / 2.')
        fullTimes[int((nPulse-1)/2)] = maxTime / 2
        fullTimes[int((nPulse-1)/2) + 1 : nPulse] = maxTime - np.flip(timePulsesHalf)
    return fullTimes

def chi(timePulses, maxTime, freqs, noiseSpectrum, weights, nPulse, symmetric):
    if symmetric==True:
        # In this case, timePulses is just the first half of the pulse sequence.
        # We will generate the rest of the sequence using function symmetricPulseTimings.
        filter_func = ps.FilterFunc(freqs, symmetricPulseTimings(timePulses, maxTime, nPulse), maxTime)
    else:
        filter_func = ps.FilterFunc(freqs, timePulses, maxTime)
    if weights is None:
        dfreq = freqs[1] - freqs[0]
        return np.sum(filter_func * noiseSpectrum * dfreq)
    else:
        return np.sum(filter_func * noiseSpectrum * weights)
 
def logchi(timePulses, maxTime, freqs, noiseSpectrum, weights, nPulse, symmetric):
     return np.log10(chi(timePulses, maxTime, freqs, noiseSpectrum, weights, nPulse, symmetric))

def band_integral(timePulses, maxTime, freqs, weights):
    """ Return negative of integral because we want to minimize it
    """
    filter_func = ps.FilterFunc(freqs, timePulses, maxTime)
    if weights is None:
            dfreq = freqs[1] - freqs[0]
            return -np.sum(filter_func * dfreq)
    else:
        return -np.sum(filter_func * weights)

def constraint_func(x):
     """
     PARAMETERS:
     x (np.ndarray): array of shape (n,)
     """
     n = len(x)
     return np.sort(x)[1:n] - np.sort(x)[0:n-1]

def constraint_bounds(x, eps, maxTime):
     """
     PARAMETERS:
     x (real np.ndarray): array of shape (n,)
     eps (scalar): epsilon, number much smaller than 1
     maxTime (scalar): maximum evolution time. Must be greater than eps
     RETURNS:
     lb, ub (real np.ndarrays): lower and upper bounds
     """
     n = len(x)
     lb = np.full(n - 1, eps)
     ub = np.full(n - 1, maxTime - eps)
     return lb, ub

def gradientFilter(freqs, timePulses, maxTime):
    """ Calculates the gradient with respect to pulse timings of filter function.
        Times must be sorted.
    """
    nFreq = len(freqs)
    nPulse = len(timePulses)
    ft = ps.fTilde(freqs, timePulses, maxTime)
    phasor = np.zeros((nPulse, nFreq), dtype=complex)
    # Build up phasor matrix row by row
    for k in range(nPulse):
        phasor[k] = (-1)**(k+1) * np.exp(1j * 2 * np.pi * freqs * timePulses[k]) * ft
    return -4 * phasor.real

def chi_gradient_exact(timePulses, maxTime, freqs, noiseSpectrum, weights, nPulse, symmetric):
    grad_filter = gradientFilter(freqs, np.sort(timePulses), maxTime)
    if weights is None:
            dfreq = freqs[1] - freqs[0]
            return np.matmul(grad_filter, (noiseSpectrum * dfreq))
    else:
        return np.matmul(grad_filter, (noiseSpectrum * weights))

def optimize_band(initialTimes, maxTime, freqs, weights, boundEps):
     
     cons = opt.NonlinearConstraint(constraint_func, *constraint_bounds(initialTimes, boundEps, maxTime))
     left_bound = np.zeros(len(initialTimes)) + boundEps
     upper_bound = np.full(len(initialTimes), maxTime) - boundEps
     bnds = opt.Bounds(lb=left_bound, ub=upper_bound)
     res = opt.minimize(band_integral, initialTimes, method='trust-constr', args=(maxTime, freqs, weights), bounds=bnds, constraints=cons)
     # Need wrapper func for SHGO due to bug in the scipy code regarding additional objective function args
     #wrapper_func = lambda a, b=maxTime, c=freqs, d=weights: band_integral(a, b, c, d)
     #res = opt.shgo(wrapper_func, bounds)
     return res

def optimize_chi(initialTimes, maxTime, freqs, noiseSpectrum, weights, boundEps, nPulse, symmetric=True):
    left_bound = np.zeros(len(initialTimes)) + boundEps
    if symmetric==True:
        upper_bound = np.full(len(initialTimes), maxTime/2) - boundEps
        cons = opt.NonlinearConstraint(constraint_func, *constraint_bounds(initialTimes, boundEps, maxTime/2))
    else:
        upper_bound = np.full(len(initialTimes), maxTime) - boundEps
        cons = opt.NonlinearConstraint(constraint_func, *constraint_bounds(initialTimes, boundEps, maxTime))
    bnds = opt.Bounds(lb=left_bound, ub=upper_bound)
    #res = opt.minimize(chi, initialTimes, method='trust-constr', args=(maxTime, freqs, noiseSpectrum, weights), constraints=cons)
    #res = opt.minimize(chi, initialTimes, method='trust-constr', jac=chi_gradient_exact, args=(maxTime, freqs, noiseSpectrum, weights), constraints=cons)
    
    #res = opt.minimize(logchi, initialTimes, method='trust-constr', args=(maxTime, freqs, noiseSpectrum, weights, nPulse, symmetric), bounds=bnds, constraints=cons)
    res = opt.minimize(chi, initialTimes, method='SLSQP', jac=chi_gradient_exact, args=(maxTime, freqs, noiseSpectrum, weights, nPulse, symmetric), bounds=bnds, constraints=cons)
    #res = opt.minimize(chi, initialTimes, method='trust-constr', args=(maxTime, freqs, noiseSpectrum, weights, nPulse, symmetric), bounds=bnds, constraints=cons)

    return res


def optimize_and_plot(nPulse, maxTime, freqs, noiseSpectrum, weights, boundEps, title,
                      save=None, show=True, symmetric=True, filterScale='linear', noiseScale='linear'):

    # Pulse times for known solutions
    udd_times = ps.UDD(nPulse, maxTime)
    cpmg_times = ps.CPMG(nPulse, maxTime)

    # OPTIMIZE!
    if symmetric==True:
        if (nPulse % 2 == 0):
            #initial_times = rpt.random_pulse_times_symmetric(int(nPulse/2), maxTime/2, boundEps)
            #initial_times = rpt.random_pulse_times(int(nPulse/2), maxTime/2, boundEps)
            initial_times = ps.CPMG(nPulse, maxTime)[0:int(nPulse/2)]
        else:
            #initial_times = rpt.random_pulse_times_symmetric(int((nPulse-1)/2), maxTime/2, boundEps)
            #initial_times = rpt.random_pulse_times(int((nPulse-1)/2), maxTime/2, boundEps)
            initial_times = ps.CPMG(nPulse, maxTime)[0:int((nPulse - 1)/2)]
    else:
        #initial_times = rpt.random_pulse_times(nPulse, maxTime, boundEps)
        initial_times = ps.CPMG(nPulse, maxTime)

    
    optim_res = optimize_chi(initial_times, maxTime, freqs, noiseSpectrum, weights, boundEps, nPulse, symmetric)#optimize_band(initial_times, maxTime, band_freqs, weights, boundEps)
    optim_times = np.sort(optim_res.x)

    # Get second half of pulse timings.
    if symmetric==True:
        optim_times = symmetricPulseTimings(optim_times, maxTime, nPulse)

    # Consolidate everything for plotter
    pulse_times = np.zeros((3, nPulse))
    pulse_times[0] = optim_times
    pulse_times[1] = udd_times
    pulse_times[2] = cpmg_times


    chi_solutions = np.zeros(3)
    chi_solutions[0] = chi(optim_times, maxTime, freqs, noiseSpectrum, weights, nPulse, False)
    chi_solutions[1] = chi(udd_times, maxTime, freqs, noiseSpectrum, weights, nPulse, False)
    chi_solutions[2] = chi(cpmg_times, maxTime, freqs, noiseSpectrum, weights, nPulse, False)

    # Print optimal chi and chi for solutions
    print("\nMinimum integral")
    print(chi_solutions[0])
    print(optim_times)
    print("\nUDD")
    print(chi_solutions[1])
    print(udd_times)
    print("\nCPMG")
    print(chi_solutions[2])
    print(cpmg_times)

    # Check if pulse is symmetric
    #check_seq_symmetric(optim_times, maxTime, 1e-4)


    filters = np.zeros((3, N_freq_pts))
    filters[0] = ps.FilterFunc(freqs, optim_times, maxTime)
    filters[1] = ps.FilterFunc(freqs, udd_times, maxTime)
    filters[2] = ps.FilterFunc(freqs, cpmg_times, maxTime)


    pt.plot_solutions(maxTime, freqs, noiseSpectrum, pulse_times,
                      nPulse, filters, chi_solutions, title,
                      save=save, show=show, filterScale=filterScale, noiseScale=noiseScale)



max_time = 1
Npulse = 4
Nsample = 10
cpmg_peak_freq = Npulse / (2 * max_time) # first peak of CPMG filter
#max_freq = Nsample / (2 * max_time)
max_freq = (4096/2 - 1) / (4096 * 1e-3)#8.0 * cpmg_peak_freq
N_time_pts = 300
N_freq_pts = 4096
time_mesh = np.linspace(0, max_time, N_time_pts + 2)[1:N_time_pts+1] # Cut off min and max time points


band_min = 1.5 * cpmg_peak_freq #max_freq/3
band_max = 2.5 * cpmg_peak_freq #max_freq/2
frequencies = np.linspace(0, max_freq, N_freq_pts)

bound_eps = 1e-3
wghts = None

# Choose noise model

#noise = ps.lorentzian(frequencies, 0, max_freq/20)#ps.lorentzian(frequencies, 0, max_freq/5) + ps.lorentzian(frequencies, max_freq/2, max_freq/7)#np.exp(-frequencies**2)
#cnb_noise = np.heaviside(frequencies - 0, 1) - np.heaviside(frequencies - band_min, 1) \
#                + np.heaviside(frequencies - band_max, 1) - np.heaviside(frequencies - max_time, 1)
#noise = cnb_noise
mu = 0.25
temp = 0.01
noise = ps.fermi_dirac(frequencies, mu, temp)

# MAKE NOISE MODEL WITH BOXES AROUND BAND AND THEN USE LOGCHI

# optimize_and_plot(Npulse, max_time, frequencies, noise, wghts, bound_eps,
#                   title='Band Noise',save=None, show=True, filterScale='linear', noiseScale='linear')
optimize_and_plot(Npulse, max_time, frequencies, noise, wghts, bound_eps,
                  title='$\mu = {0}, T = {1}$'.format(mu, temp),save=None, show=True, filterScale='linear', noiseScale='linear')

# Ntemp = 20
# min_temp = 0.001
# max_temp = 1
# mu = 0#cpmg_peak_freq / 3
# temps = np.linspace(min_temp, max_temp, Ntemp)
# directory = 'temp_sweep'
# for i in range(Ntemp):
#     file = 'job_{:03d}'.format(i)
#     path = os.path.join(directory, file)
#     noise = ps.fermi_dirac(frequencies, mu, temps[i])
#     job_title = '$\mu = {0}, T = {1}$'.format(mu, temps[i])
#     optimize_and_plot(Npulse, max_time, frequencies, noise, wghts, bound_eps,
#                       title=job_title, save=path, show=False, filterScale='log', noiseScale='linear', symmetric=False)


"""
NOTES:
- For sum of Lorentzians, optimal solution is symmetric. Not sure if local or global because cannot visualize 4D.
- For Fermi-Dirac hard cutoff, same thing as above lines.

TO-DO:
- Make initial guess random?
"""
