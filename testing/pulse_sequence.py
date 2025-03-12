import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define UDD and CPMG sequences

def UDD(Npulse, maxTime):
    pulse_indices = np.arange(1, Npulse + 1) # size Npulse. Final value is Npulse
    deltas = np.sin(np.pi * pulse_indices / (2 * Npulse + 2))**2
    return deltas * maxTime

def PDD(Npulse, maxTime):
    pulse_indices = np.arange(1, Npulse + 1) # size Npulse. Final value is Npulse
    deltas = pulse_indices / (Npulse + 1) 
    return deltas * maxTime

def CPMG(Npulse, maxTime):
    pulse_indices = np.arange(1, Npulse + 1) # size Npulse. Final value is Npulse
    deltas = (2 * pulse_indices - 1) / (2 * Npulse) 
    return deltas * maxTime

def FID(maxTime):
    return 

def CPMGPeakFreq(Npulse, maxTime):
    """
    The first peak frequency of the CPMG filter function
    """
    return Npulse / (2 * maxTime)

def FIDFilter(freqs, maxTime):
    return maxTime**2 * np.sinc(freqs * maxTime)**2

def fTilde(freqs, timePulses, maxTime):
    """
    PARAMETERS:
    freqs (np.ndarray, 1D): frequencies in Hertz. Must be ordered from least to greatest
    timePulses (np.ndarray, 1D): times at which pulses are applied, sorted from least to greatest, must be greater than 0
    RETURNS:
    FourierTrans_ft (np.ndarray, 1D): Fourier transform of f(t) function 
    """
    Np = len(timePulses)
    Nf = len(freqs)
    
    timeWithEnds = np.zeros(Np+2)
    timeWithEnds[0] = 0
    timeWithEnds[-1] = maxTime
    timeWithEnds[1:Np+1] = timePulses

    a = timeWithEnds[1:] - timeWithEnds[:Np+1]
    b = (timeWithEnds[1:] + timeWithEnds[:Np+1]) / 2
    
    FourierTrans_ft = np.zeros(Nf, dtype='complex')
    
    # Make more efficient using vectorization?
    for n in range(Np+1):
        FourierTrans_ft += (-1)**n * np.exp(-1j * 2 * np.pi * freqs * b[n]) * a[n] * np.sinc(freqs * a[n])
    
    # Resize arrays into sparse mesh encoding. Similar to numpy's meshgrid.
    #amesh, bmesh = np.array([a]), np.array([b])
    #freqsmesh = np.array([freqs]).T
    #pm_ones = np.array([(-1)**np.arange(Np+1)]) # Array of plus or minus ones
    #FourierTrans_ft = np.sum(pm_ones * np.exp(-1j * 2 * np.pi * freqsmesh * bmesh) * amesh * np.sinc(freqsmesh * amesh), axis = 1)
    

    #if timePulses.ndim == 2:
    #    Nf = len(freqs)
    #    Nseq = timePulses.shape[0]
    #    Np = timePulses.shape[1] # number of pulses in each pulse sequence
    #    zeros = np.zeros((Nseq, 1))
    #    maxtimes = np.full((Nseq, 1))
    #    timeWithEnds = np.hstack((zeros, timePulses))
    #    timeWithEnds = np.hstack((timePulses, maxtimes))

    #    a = timeWithEnds[:, 1:] - timeWithEnds[:, :Np+1]
    #    b = (timeWithEnds[:, 1:] - timeWithEnds[:, :Np+1]) / 2

    #    FourierTrans_ft = np.zeros((Nseq, Nf), dtype='complex')

    return FourierTrans_ft

def FilterFunc(freqs, timePulses, maxTime):
    """
    PARAMETERS:
    freqs (np.ndarray, 1D): frequencies in Hertz. Must be ordered from least to greatest
    timePulses (np.ndarray, 1D): times at which pulses are applied, sortted from least to greatest
    tol (float): used to determine when to use Taylor expansion of Filter function around angular freq * time = 0
    RETURNS:
    FF (np.ndarray, 1D): filter function for each frequency
    """
    FourierTrans_ft = fTilde(freqs, timePulses, maxTime) 
    FF = ( FourierTrans_ft * FourierTrans_ft.conj() ).real # take real to convert array to real data type
    
    return FF

def randomSequence(nPulse, maxTime, nSequence=None):
    rng = np.random.default_rng() # Random number generator
    if nSequence is not None:
        times = rng.uniform(0, maxTime, size=(nSequence, nPulse))
        return np.sort(times, axis=1)
    else:
        times = rng.uniform(0, maxTime, size=nPulse)
        return np.sort(times)

def optimaWidths(x, f):
    """
    Calculate the width of optima in an array representing function values.
    PARAMETERS:
    x (np.ndarray): uniformly spaced abscissa
    f (np.ndarray): uniformly spaced ordinate
    RETURNS:
    crest_widths (np.ndarray), trough_widths (np.ndarray): widths of the crests (maxima) and troughs (minima)
    """
    if x.ndim != 1:
        raise ValueError("x is not of dimension 1.")
    if f.ndim != 1:
        raise ValueError("f is not of dimension 1.")
    if len(x) != len(f):
        raise ValueError("x and f are not of the same size")
    nPts = len(x)
    crest_widths = []
    trough_widths = []
    for i in range(nPts-2):
        left_idx = i
        mid_idx = i + 1
        right_idx = i + 2
        # Find a crest (aka maximum).
        # Take three points. If the middle point is greater than the left and right points:
        if f[left_idx] < f[mid_idx] and f[mid_idx] > f[right_idx]:
            # Calculate width of crest.
            crest_w = x[right_idx] - x[left_idx]
            # If left_idx or right_idx already have their min or max possible values, respectively,
            # append the width to the list and go to next for loop iteration.
            if left_idx == 0 or right_idx == nPts - 1:
                crest_widths.append(crest_w)
                continue
            # While left index is greater than min index and right index is less than max index.
            while 0 < left_idx and right_idx < nPts - 1:
                # If next left and right points are less than the previous ones
                if f[left_idx + 1] < f[left_idx] and f[right_idx] > f[right_idx + 1]:
                    left_idx -= 1
                    right_idx += 1
                    crest_w = x[right_idx] - x[left_idx]
                else:
                    crest_widths.append(crest_w)
                    break # End while loop
        # Find a trough (aka minimum).
        # Take three points. If the middle point is less than the left and right points:
        elif f[left_idx] > f[mid_idx] and f[mid_idx] < f[right_idx]:
            # Calculate width of trough.
            trough_w = x[right_idx] - x[left_idx]
            # If left_idx or right_idx already have their min or max possible values, respectively,
            # append the width to the list and go to next for loop iteration.
            if left_idx == 0 or right_idx == nPts - 1:
                trough_widths.append(trough_w)
                continue
            # While left index is greater than min index and right index is less than max index.
            while 0 < left_idx and right_idx < nPts - 1:
                # If next left and right points are greater than the previous ones
                if f[left_idx + 1] > f[left_idx] and f[right_idx] < f[right_idx + 1]:
                    left_idx -= 1
                    right_idx += 1
                    trough_w = x[right_idx] - x[left_idx]
                else:
                    trough_widths.append(trough_w)
                    break # End while loop
        
    crest_widths, trough_widths = np.ndarray(crest_widths), np.ndarray(trough_widths)
    return crest_widths, trough_widths

def temp_from_cutoff(chem_potential, cutoff_freq, cutoff_noise = 1e-6):
    return (cutoff_freq - chem_potential) / np.log( 1/cutoff_noise - 1 )

def fermi_dirac(freqs, chem_potential, temperature):
    return 1 / (np.exp( (freqs - chem_potential) / temperature ) + 1)

def one_over_f_noise(freqs, epsilon=1e-6):
    # Calculate reciprocal only if element is greater than epsilon
    recip = np.reciprocal(freqs, where=freqs >= epsilon)
    # If elements are less than epsilon, replace with 1/epsilon. Else, keep original value.
    noise = np.where(recip < epsilon, 1/epsilon, recip)
    return noise

def lorentzian(abscissa, center, fwhm):
    """ Normalized Lorentzian function evaluated at the value(s) specified by
        abscissa.

        PARAMETERS:
        abscissa (number of np.ndarray): x axis values to evaluate f(x)
        center (number): x axis value at which function is peaked
        fwhm (number): full width at half maximum

        RETURNS:
        number or np.ndarray
    """
    numerator = fwhm / 2 / np.pi
    denominator = (abscissa - center)**2 + (fwhm / 2)**2
    return numerator / denominator

def lorentzians(abscissa, centers, fwhms):
    """ Sum of normalized Lorentzian functions evaluated on the value(s) specified by
        abscissa. Each function has its own center and full width at half maximum.
        
        PARAMETERS:
        abscissa (number of np.ndarray): x axis values to evaluate f(x)
        centers (np.ndarray): x axis values at which function is peaked
        fwhms (np.ndarray): full widths at half maximums for each value of centers

        RETURNS:
        sum (number or np.ndarray)
    """
    if len(centers) != len(fwhms):
        raise ValueError('The np.ndarrays centers and fwhms must have dimension 1 and must have the same size.')
    sum = 0
    for i in range(len(centers)):
        sum += lorentzian(abscissa, centers[i], fwhms[i])
    return sum

def chi(freqs, noise, filter_func, weights=None):
    """
    freqs (np.ndarray): frequencies in units of 1/time
    noise (np.ndarray): noise spectrum
    filter_func (np.ndarray): filter function calculated with the ordinary frequency, unitary Fourier transform convention
    """
    if weights is None:
        dfreq = freqs[1] - freqs[0]
        return np.sum(filter_func * noise) * dfreq
    else:
        return np.sum(filter_func * noise * weights)

def chi_avg(freqs, noise, maxTime, nPulseChances, weights=None):
    pulse_spacing = maxTime / (nPulseChances + 1)
    ff = (nPulseChances + 1) * (pulse_spacing * np.sinc(freqs * pulse_spacing))**2
    return chi(freqs, noise, ff, weights=weights)

def fidelity(chi_array):
    return 0.5 * (1 + np.exp(-chi_array))

def RewardFunc(chi_array, initial_chi = 1.0):
    #small = 1e-8
    #rewards = np.zeros_like(chi_array)
    #for i, x in np.ndenumerate(chi_array):
    #    if x < 0:
    #        rewards[i] = 1 / small
    #    elif x < small:
    #        rewards[i] = 1 / small
    #    else:
    #        rewards[i] = 1 / x
    #
    #return initial_chi / chi_array
    #return 100 * 0.5 * (1 + np.exp( -chi_array ) )
    #
    #avgInfid = 1 - 0.5 * (1 + np.exp( -chi_array ) )
    #initialAvgInfid = 1 - 0.5 * np.full_like(avgInfid, 1 + np.exp(-initial_chi))
    #relativeAvgInfid = avgInfid / initialAvgInfid
    #return 1 / (relativeAvgInfid + 1e-8)
    avgFid = fidelity(chi_array)
    return avgFid / (1 - avgFid + 1e-8)

def fid_from_reward(reward):
    return reward * (1 + 1e-8) / (reward + 1)

def calc_sign_seq(pulse_seq, Nb=1):
    """ Given a pulse sequence of zeros and ones, calculates the
        value of f(t) in the time between pulse chances, as well
        as in the times between time zero and the first pulse chance,
        and the time between time tmax and the last pulse chance.
        
        Example where x represents beginnning/end time, 0s and 1s are
        the pulse sequence values, and the +-1s are the values of f(t):
        pulse sequence. Here, Nc=4, Nb=3.
        x     -    0    -    1    -    0    -    1    -    x
        f(t):
             111       111     -1-1-1    -1-1-1      111
    """
    Nc = len(pulse_seq) # Number of pulse chances
    #sign_seq = np.ones(Npc + 1)
    sign_seq = np.ones((Nc + 1) * Nb)

    sign = 1
    for i in range(Nc):
        if pulse_seq[i] > 0:
            sign = -sign
        idx1 = Nb * (i + 1)
        idx2 = Nb * (i + 2) - 1
        #sign_seq[i+1] = sign
        sign_seq[idx1 : idx2 + 1] *= sign # +1 because of numpy slicing
    return sign_seq

def cutoff_index(v, S):
    """
    PARAMETERS:
    v (np.ndarray): frequency in units 1/time
    S (np.ndarray): noise spectrum, assumed to be asymptotically decaying 
    RETURNS:
    cutoffIdx (int): index to cutoff noise
    """
    if v.ndim != 1:
        raise ValueError("v is not of dimension 1.")
    if S.ndim != 1:
        raise ValueError("S is not of dimension 1.")
    if len(v) != len(S):
        raise ValueError("v and S are not of the same size")

    cutoffEps = 1e-6
    nPts = len(v)
    cutoffIdx = nPts - 1 # Initialize to max index
    for i in range(cutoffIdx, -1, -1):
        if (S[i] / v[i]**2 > cutoffEps):
            cutoffIdx = i
            break
    return cutoffIdx
