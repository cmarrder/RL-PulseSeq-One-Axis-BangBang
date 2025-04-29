import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import sici
import plot_tools as pt
import gaussianquad as gq

import sys
import time

def print_progress_bar(iteration, total, prefix='Progress', suffix='Complete', length=30, fill='█'):
    """
    Prints a progress bar to the console.

    Args:
        iteration (int): Current iteration number.
        total (int): Total number of iterations.
        prefix (str, optional): String to display before the bar. Defaults to ''.
        suffix (str, optional): String to display after the bar. Defaults to ''.
        length (int, optional): Character length of the bar. Defaults to 30.
        fill (str, optional): Character to use for the filled portion. Defaults to '█'.
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    return


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

def FilterFunc(freqs, timePulses, maxTime, method = 1):
    """
    PARAMETERS:
    freqs (np.ndarray, 1D): frequencies in Hertz. Must be ordered from least to greatest
    timePulses (np.ndarray, 1D): times at which pulses are applied, sortted from least to greatest
    tol (float): used to determine when to use Taylor expansion of Filter function around angular freq * time = 0
    RETURNS:
    FF (np.ndarray, 1D): filter function for each frequency
    """
    if method == 1:
        FourierTrans_ft = fTilde(freqs, timePulses, maxTime) 
        FF = ( FourierTrans_ft * FourierTrans_ft.conj() ).real # take real to convert array to real data type
    elif method == 2:
        Np = len(timePulses)
        Nf = len(freqs)
        
        timeWithEnds = np.zeros(Np+2)
        timeWithEnds[0] = 0
        timeWithEnds[-1] = maxTime
        timeWithEnds[1:Np+1] = timePulses

        a = timeWithEnds[1:] - timeWithEnds[:Np+1]
        b = (timeWithEnds[1:] + timeWithEnds[:Np+1]) / 2

        FF = np.zeros(Nf)
        for n in range(Np + 1):
            FF += a[n]**2 * np.sinc(freqs * a[n])**2
        for i in range(Np):
            for j in range(i + 1, Np + 1):
                FF += 2 * (-1)**(i + j) * a[i] * a[j] * np.cos(2 * np.pi * freqs * (b[i] - b[j])) * np.sinc(freqs * a[i]) * np.sinc(freqs * a[j])
    elif method == 3:
        Np = len(timePulses)
        Nf = len(freqs)
        
        timeWithEnds = np.zeros(Np+2)
        timeWithEnds[0] = 0
        timeWithEnds[-1] = maxTime
        timeWithEnds[1:Np+1] = timePulses

        alphas = np.ones(Np + 2)
        alphas[1:-1] = 2

        FF = np.zeros(Nf)

        for n in range(Np + 1):
            for m in range(n + 1, Np + 2):
                FF += alphas[n] * alphas[m] * (-1)**(n + m) * np.cos(2 * np.pi * freqs * (timeWithEnds[n] - timeWithEnds[m]))

        FF += Np + 2
        FF /= (2 * np.pi**2 * freqs**2)
    
    return FF

def FilterFuncMaclaurin(freqs, timePulses, maxTime):
    """
    8th order Maclaurin series for filter function
    """
    Np = len(timePulses)
    Nf = len(freqs)
    
    timeWithEnds = np.zeros(Np+2)
    timeWithEnds[0] = 0
    timeWithEnds[-1] = maxTime
    timeWithEnds[1:Np+1] = timePulses
    
    a = timeWithEnds[1:] - timeWithEnds[:Np+1]
    b = (timeWithEnds[1:] + timeWithEnds[:Np+1]) / 2
    
    FF = np.zeros(Nf)
    x = np.pi * freqs
    for n in range(Np + 1):
        # Series expansion!
        term1 = 1 + (1/3)*(a[n]*x)**2 + (2/45)*(a[n]*x)**4 - (1/315)*(a[n]*x)**6 + (2/14175)*(a[n]*x)**8
        FF += a[n]**2 * term1
    for i in range(Np):
        for j in range(i + 1, Np + 1):
            d = b[i] - b[j]
            # Define series coefficients
            order2 = - (a[i]**2 + a[j]**2 + 12*d**2) / 6
            order4 = (3*a[i]**4 + 10*a[i]**2 * (a[j]**2 + 12*d**2) + 3*(a[j]**4 + 40*(a[j]*d)**2 + 80*d**4)) / 360
            order6 = (-a[i]**6 - 7*a[i]**4 * (a[j]**2 + 12*d**2) - 7*a[i]**2 * (a[j]**4 + 40*a[j]**2 * d**2 + 80*d**4) \
                    - a[j]**6 - 84*a[j]**4 * d**2 - 560*a[j]**2 * d**4 - 448*d**6) / 5040
            order8 = ( 5*a[i]**8 + 60*a[i]**6 * (a[j]**2 + 12*d**2) + 126*a[i]**4 * (a[j]**4 + 40*(a[j]*d)**2 + 80*d**4) \
                    + 60 * a[i]**2 * (a[j]**6 + 84*a[j]**4 * d**2 + 560*a[j]**2 * d**4 + 448*d**6) \
                    + 5 * (a[j]**2 + 12*d**2) * (a[j]**6 + 132*a[j]**4 * d**2 + 432*a[j]**2 * d**4 + 192*d**6) ) / 1814400
            # Series expansion!
            term2 = 1 + order2 * x**2 + order4 * x**4 + order6 * x**6 + order8 * x**8
            FF += 2 * (-1)**(i + j) * a[i] * a[j] * term2 
    return FF

def calculateT2star(cutoff_freq, noise_func, maxTimes, verbose=False, plot=False):
    # This comes from the fact that if v is frequency,
    # the zeros of sinc(v*T) = sin(pi*v*T)/(pi*v*T)
    # are approximately 1/T apart in frequency space.
    freq_resolutions = 1 / maxTimes

    if verbose:
        print("In calculateT2star:")

    # Array storing chi values
    chis = np.zeros_like(maxTimes)
    for i in range(len(maxTimes)):
        nFreq = 20 * int(cutoff_freq / freq_resolutions[i])
        if verbose:
            print("i = {0}, T = {1}, nFreq = {2}".format(i, maxTimes[i], nFreq))
        freqs, weights = gq.legendreQuad5(0, cutoff_freq, int(nFreq/5))
        noise = noise_func(freqs)
        # Filter function for free induction decay
        ff = (maxTimes[i] * np.sinc(freqs * maxTimes[i]))**2
        chis[i] = chi(freqs, noise, ff, weights=weights)

    # Polynomial fitting
    max_order = 10
    pc = np.polyfit(maxTimes, chis, max_order)
    best_order_coeff = np.max(pc)
    best_order_index = np.argmax(pc)
    best_order = max_order - best_order_index

    print("\nOrder of best polynomial fit: {}".format(best_order))

    # Make sure best_order_coeff is much bigger than the higher order coefficients
    for j in range(best_order_index):
        coeff = pc[j]
        if abs(coeff / best_order_coeff) > 1e-3:
            print("\nWarning: other polynomial fit coefficients are comparable to the largest coefficient.")
            print("Order: {}".format(max_order - j))
            print("|coefficient / best_order_coeff| = {}".format(abs(coeff / best_order_coeff)))

    if best_order == 0:
        T2star = best_order_coeff
    else:
        T2star = (best_order_coeff)**(-best_order)

    if plot:
        plt.plot(maxTimes, chis)
        plt.xlabel("$T$")
        plt.ylabel("$\chi$")
        plt.show()
        plt.close()
    return T2star

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
                if f[left_idx - 1] < f[left_idx] and f[right_idx] > f[right_idx + 1]:
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
                if f[left_idx - 1] > f[left_idx] and f[right_idx] < f[right_idx + 1]:
                    left_idx -= 1
                    right_idx += 1
                    trough_w = x[right_idx] - x[left_idx]
                else:
                    trough_widths.append(trough_w)
                    break # End while loop
        
    crest_widths, trough_widths = np.array(crest_widths), np.array(trough_widths)
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

def lorentzian(abscissa, center, fwhm, height=None):
    """ Lorentzian function evaluated at the value(s) specified by
        abscissa. If height is not specified, returns normalized Lorentzian.

        PARAMETERS:
        abscissa (number of np.ndarray): x axis values to evaluate f(x)
        center (number): x axis value at which function is peaked
        fwhm (number): full width at half maximum
        height (number): height of Lorentzian

        RETURNS:
        number or np.ndarray
    """
    if height is None:
        numerator = fwhm / 2 / np.pi
        denominator = (abscissa - center)**2 + (fwhm / 2)**2
    else:
        numerator = height * (fwhm/2)**2
        denominator = (abscissa - center)**2 + (fwhm / 2)**2
    return numerator / denominator

def lorentzians(abscissa, centers, fwhms, heights=None):
    """ Sum of normalized Lorentzian functions evaluated on the value(s) specified by
        abscissa. Each function has its own center and full width at half maximum.
        If height is not specified, returns sum of Lorentzians, each of which is normalized.
        
        PARAMETERS:
        abscissa (number of np.ndarray): x axis values to evaluate f(x)
        centers (np.ndarray): x axis values at which function is peaked
        fwhms (np.ndarray): full widths at half maximums for each value of centers
        heights (np.ndarray): heights for each Lorentzian peak

        RETURNS:
        sum (number or np.ndarray)
    """
    sum = 0
    if heights is None:
        if len(centers) != len(fwhms):
            raise ValueError('The np.ndarrays centers and fwhms must have dimension 1 and must have the same size.')
        for i in range(len(centers)):
            sum += lorentzian(abscissa, centers[i], fwhms[i])
    else:
        if len(centers) != len(fwhms) and len(centers) != len(heights):
            raise ValueError('The np.ndarrays centers, fwhms, and heights must have dimension 1 and must have the same size.')
        for i in range(len(centers)):
            sum += lorentzian(abscissa, centers[i], fwhms[i], height = heights[i])
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

def chi_const_noise(timePulses, maxTime, constant, freq_cutoff):
    """ Calculate chi when noise S is a constant from frequency (1/time) 0 to freq_cutoff.
    PARAMETERS:
    constant (double)
    freq_cutoff (double)
    time_pulses (np.ndarray, 1D): times at which pulses are applied, sortted from least to greatest
    maxTime (double): max evolution time
    """

    Np = len(timePulses)
    
    timeWithEnds = np.zeros(Np+2)
    timeWithEnds[0] = 0
    timeWithEnds[-1] = maxTime
    timeWithEnds[1:Np+1] = timePulses
    
    # angular frequency
    angfreq_cutoff = 2 * np.pi * freq_cutoff

    def x_Si(x):
        return x * sici(x)[0]

    sum1 = 0
    sum2 = 0
    for n in range(Np + 1):
        an = timeWithEnds[n+1] - timeWithEnds[n]
        phase = angfreq_cutoff * an
        sum1 += (x_Si(phase) + np.cos(phase) - 1) / angfreq_cutoff
    for i in range(Np):
        for j in range(i+1, Np+1):
            sign = (-1)**(i+j)
            # Define time differences for calculation
            phi1 = angfreq_cutoff * (timeWithEnds[j] - timeWithEnds[i+1])
            phi2 = angfreq_cutoff * (timeWithEnds[j+1] - timeWithEnds[i])
            phi3 = angfreq_cutoff * (timeWithEnds[j] - timeWithEnds[i])
            phi4 = angfreq_cutoff * (timeWithEnds[j+1] - timeWithEnds[i+1])

            sum2 += sign * \
                    (x_Si(phi1) + x_Si(phi2) \
                    - x_Si(phi3) - x_Si(phi4) \
                    + np.cos(phi1) + np.cos(phi2) - np.cos(phi3) - np.cos(phi4))
    return constant * (sum1 + sum2 / angfreq_cutoff) / np.pi

def chi_1_over_f_noise(timePulses, maxTime, amplitude, freq_cutoff):
    Np = len(timePulses)
    
    timeWithEnds = np.zeros(Np+2)
    timeWithEnds[0] = 0
    timeWithEnds[-1] = maxTime
    timeWithEnds[1:Np+1] = timePulses
    
    angfreq_cutoff = 2 * np.pi * freq_cutoff

    def xSq_Ci(x):
        if x == 0:
            return 0
        else:
            return x**2 * sici(x)[1]

    sum3 = 0
    sum4 = 0
    for n in range(Np + 1):
        an = timeWithEnds[n+1] - timeWithEnds[n]
        phase = angfreq_cutoff * an
        sum3 += an**2 * (-sici(phase)[1] + 0.5 * np.sinc(an * freq_cutoff)**2 + np.sinc(2 * an * freq_cutoff))
    for i in range(Np):
        for j in range(i+1, Np+1):
            sign = (-1)**(i+j)
            # Define time differences for calculation
            phi1 = angfreq_cutoff * (timeWithEnds[j] - timeWithEnds[i+1])
            phi2 = angfreq_cutoff * (timeWithEnds[j+1] - timeWithEnds[i])
            phi3 = angfreq_cutoff * (timeWithEnds[j] - timeWithEnds[i])
            phi4 = angfreq_cutoff * (timeWithEnds[j+1] - timeWithEnds[i+1])

            sum4 += sign * \
                    ( - xSq_Ci(phi1) - xSq_Ci(phi2) \
                    + xSq_Ci(phi3) + xSq_Ci(phi4) \
                    + phi1 * np.sin(phi1) + phi2 * np.sin(phi2) \
                    - phi3 * np.sin(phi3) - phi4 * np.sin(phi4) \
                    - np.cos(phi1) - np.cos(phi2) + np.cos(phi3) + np.cos(phi4))
    return amplitude * (sum3 + sum4 / angfreq_cutoff**2)

def chi_1_over_f_IR_cutoff(timePulses, maxTime, amplitude, freq_cutoff):
    """
    freq_cutoff is the infrared (IR) cutoff
    """
    Np = len(timePulses)
    
    timeWithEnds = np.zeros(Np+2)
    timeWithEnds[0] = 0
    timeWithEnds[-1] = maxTime
    timeWithEnds[1:Np+1] = timePulses
    
    angfreq_cutoff = 2 * np.pi * freq_cutoff

    def x_Si(x):
        return x * sici(x)[0]
    def xSq_Ci(x):
        if x == 0:
            return 0
        else:
            return x**2 * sici(x)[1]

    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for n in range(Np + 1):
        an = timeWithEnds[n+1] - timeWithEnds[n]
        phase = angfreq_cutoff * an
        sum1 += x_Si(phase) + np.cos(phase) - 1
        sum3 += an**2 * (-sici(phase)[1] + 0.5 * np.sinc(an * freq_cutoff)**2 + np.sinc(2 * an * freq_cutoff))
    for i in range(Np):
        for j in range(i+1, Np+1):
            sign = (-1)**(i+j)
            # Define time differences for calculation
            phi1 = angfreq_cutoff * (timeWithEnds[j] - timeWithEnds[i+1])
            phi2 = angfreq_cutoff * (timeWithEnds[j+1] - timeWithEnds[i])
            phi3 = angfreq_cutoff * (timeWithEnds[j] - timeWithEnds[i])
            phi4 = angfreq_cutoff * (timeWithEnds[j+1] - timeWithEnds[i+1])

            sum2 += sign * \
                    (x_Si(phi1) + x_Si(phi2) \
                    - x_Si(phi3) - x_Si(phi4) \
                    + np.cos(phi1) + np.cos(phi2) - np.cos(phi3) - np.cos(phi4))
            sum4 += sign * \
                    ( - xSq_Ci(phi1) - xSq_Ci(phi2) \
                    + xSq_Ci(phi3) + xSq_Ci(phi4) \
                    + phi1 * np.sin(phi1) + phi2 * np.sin(phi2) \
                    - phi3 * np.sin(phi3) - phi4 * np.sin(phi4) \
                    - np.cos(phi1) - np.cos(phi2) + np.cos(phi3) + np.cos(phi4))
    return amplitude * ((sum1 + sum2) / (np.pi * angfreq_cutoff) + sum3 + sum4 / angfreq_cutoff**2)

def test_exact_chi(freqs, noise, Np, max_time, func_exact_chi, *args, weights=None):
    Nsequence = 50
    pulse_seqs = randomSequence(Np, max_time, Nsequence)
    chi_numerical = np.zeros(Nsequence)
    chi_exact = np.zeros(Nsequence)
    for i in range(Nsequence):
        # Make random pulse sequence and calculate associated filter function
        ff = FilterFunc(freqs, pulse_seqs[i], max_time)
        chi_numerical[i] = chi(freqs, noise, ff, weights=weights)
        chi_exact[i] = func_exact_chi(pulse_seqs[i], max_time, *args)

    print("chi_numerical")
    print(chi_numerical)
    print("chi_exact")
    print(chi_exact)

    plt.scatter(range(Nsequence), np.abs(chi_numerical - chi_exact) / chi_numerical)
    plt.show()

    return


def fidelity(chi_array):
    return 0.5 * (1 + np.exp(-chi_array))

def RewardFunc(chi_array, initial_chi = 1.0):
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

def compareFilterMethods(freqs, timePulses, maxTime):
    """
    Plot the different methods for calculating the filter function.
    """
    filters = np.zeros((3, len(freqs)))
    for i in range(3):
        filters[i] = FilterFunc(freqs, timePulses, maxTime, method=i+1)
    plt.plot(freqs, np.abs(filters[0] - filters[1]) / filters[0], label='Method {}'.format(2))
    plt.plot(freqs, np.abs(filters[0] - filters[2]) / filters[0], label='Method {}'.format(3))
    plt.ylabel('$|F_{true} - F_{alternative}| / F_{true}$')
    plt.xlabel(r'$\nu$')
    plt.yscale('log')
    plt.legend()
    plt.show()
    return

def compareFilterWithSeries(freqs, nPulse, maxTime):
    nSequence = 5
    sequences = randomSequence(nPulse, maxTime, nSequence = nSequence)
    for i in range(nSequence):
        ff = FilterFunc(freqs, sequences[i], maxTime)
        ff_series = FilterFuncMaclaurin(freqs, sequences[i], maxTime)
        plt.plot(freqs, np.abs(ff - ff_series) / ff)
    plt.xlabel(r'$\nu$')
    plt.ylabel('$|F_{true} - F_{series}| / F_{true}$')
    plt.yscale('log')
    plt.show()
    return

def filterOptimalWidths(cutoffFreq, nPulse, maxTime, nSequence, nFreq):
    """ Sample many random filter functions and find the smallest and largest
        crest and trough widths.
    """
    freq = np.linspace(0, cutoffFreq, nFreq)

    sequences = randomSequence(nPulse, maxTime, nSequence = nSequence)
    min_crest_w = 1e32
    min_trough_w = 1e32
    max_crest_w = 0
    max_trough_w = 0

    min_crest_times = np.zeros(nPulse)
    min_trough_times = np.zeros(nPulse)
    max_crest_times = np.zeros(nPulse)
    max_trough_times = np.zeros(nPulse)

    for i in range(nSequence):
        ff = FilterFunc(freq, sequences[i], maxTime)
        crest_w, trough_w = optimaWidths(freq, ff)
        
        # Get the current minimum and maximum crest and trough widths
        current_min_cw, current_min_tw = np.min(crest_w), np.min(trough_w)
        current_max_cw, current_max_tw = np.max(crest_w), np.max(trough_w)

        if current_min_cw < min_crest_w:
            min_crest_w = current_min_cw
            min_crest_times = sequences[i]
        if current_min_tw < min_trough_w:
            min_trough_w = current_min_tw
            min_trough_times = sequences[i]
        if current_max_cw > max_crest_w:
            max_crest_w = current_max_cw
            max_crest_times = sequences[i]
        if current_max_tw > max_trough_w:
            max_trough_w = current_max_tw
            max_trough_times = sequences[i]

        print_progress_bar(i, nSequence)

    print('')
    print("Minimum crest width: {}".format(min_crest_w))
    print("Minimum trough width: {}".format(min_trough_w))
    print("Maximum crest width: {}".format(max_crest_w))
    print("Maximum trough width: {}".format(max_trough_w))
    print('')
    print('Pulse Times')
    return min_crest_times, min_trough_times, max_crest_times, max_trough_times

def findCutoffFreq(initial_guess, dfreq, noise_func):
    """ Find cutoff frequency for the integral \int_{0}^{cutoff} filter * noise dfreq.
        Assumes that for frequencies greater than initial_guess, noise_func is monotonically
        decreasing.
    """
    freq = initial_guess
    init_noise_div_freqSq = noise_func(freq) / freq**2
    while (noise_func(freq) / freq**2) / init_noise_div_freqSq > 1e-9:
        freq += dfreq
    return freq


if __name__=='__main__':
    max_time = 1
    cutoff_freq = 100
    S0 = 3#34.7
    nPulse = 8#64
    nFreq = 20000
    nSequence = nPulse * 10000
    
    #freqs = np.linspace(0, cutoff_freq, 5000000)
    #noise = np.full_like(freqs, S0)

    #freqs = np.linspace(cutoff_freq, 100 * cutoff_freq, 1000000)
    #noise = S0 / freqs


    IR_cutoff = 1 / max_time
    noise_func = lambda x: S0 / np.where(x <= IR_cutoff, 1, x)
    cutoff_freq = findCutoffFreq(IR_cutoff, IR_cutoff / 100, noise_func)

    #freqs, weights = gq.legendreQuad5(0, cutoff_freq, int(nFreq/5))
    #noise = noise_func(freqs)

    #plt.plot(freqs, noise)
    #plt.show()

    #test_exact_chi(freqs, noise, 2, max_time, chi_1_over_f_IR_cutoff, S0, IR_cutoff, weights=weights)

    maxTimes = np.linspace(1, 10, 100)
    T2star = calculateT2star(cutoff_freq, noise_func, maxTimes, verbose=True)
    print("T2star = ", T2star)


    #pulse_times = PDD(8, max_time)
    #ff = FilterFunc(freqs, pulse_times, max_time)
    #chi_numerical = chi(freqs, noise, ff)

    #compareFilterMethods(freqs, times, max_time)


    #freqs = np.linspace(0, cutoff_freq, 100000)
    #ff = np.sin(freqs)

    #crest_w, trough_w = optimaWidths(freqs, ff)


    #print('cw')
    #print(crest_w)
    #print('tw')
    #print(trough_w)

    #print('2 * pi')
    #print(np.pi * 2)
    """

    min_crest_times, min_trough_times, max_crest_times, max_trough_times = filterOptimalWidths(cutoff_freq, nPulse, max_time, nSequence, nFreq)

    print('Min Crest Times')
    print(min_crest_times)
    print('Min Trough Times')
    print(min_trough_times)
    print('Max Crest Times')
    print(max_crest_times)
    print('Max Trough Times')
    print(max_trough_times)


    min_crest_filter = FilterFunc(freqs, min_crest_times, max_time)
    min_trough_filter = FilterFunc(freqs, min_trough_times, max_time)
    max_crest_filter = FilterFunc(freqs, max_crest_times, max_time)
    max_trough_filter = FilterFunc(freqs, max_trough_times, max_time)

    # Make plot
    sequence_array = np.stack((min_crest_times, min_trough_times, max_crest_times, max_trough_times))
    sequence_labels = ["Times w/ Min Crest", "Times w/ Min Trough", "Times w/ Max Crest", "Times w/ Max Trough"]
    filter_array = np.stack((min_crest_filter, min_trough_filter, max_crest_filter, max_trough_filter))
    freq_label = r"$\nu$"
    filter_colors = ["blue" for i in range(4)]
    filter_linestyle = "solid"
    filter_labels = [r"$F(\nu)$" for i in range(4)] 
    noise_color = "red"
    noise_linestyle = "solid"
    noise_label = r"$S(\nu)$"
    filter_noise_titles = ["Filter w/ Min Crest", "Filter w/ Min Trough", "Filter w/ Max Crest", "Filter w/ Max Trough"]


    pt.plot_times_and_filters(sequence_array, sequence_labels,
                           max_time, freqs, freq_label,
                           filter_array, filter_colors, filter_linestyle, filter_labels,
                           noise, noise_color, noise_linestyle, noise_label,
                           filter_noise_titles, filter_noise_scale='log')
    """
