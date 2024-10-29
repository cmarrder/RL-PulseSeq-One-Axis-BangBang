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

def fTilde(freqs, timePulses, maxTime):
    """
    PARAMETERS:
    freqs (np.ndarray, 1D): frequencies in Hertz. Must be ordered from least to greatest
    timePulses (np.ndarray, 1D): times at which pulses are applied, sortted from lest to greatest
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
    FF = np.zeros(Nf)
    
    # Make more efficient using vectorization?
    for n in range(Np+1):
        FourierTrans_ft += (-1)**n * np.exp(-1j * 2 * np.pi * freqs * b[n]) * a[n] * np.sinc(freqs * a[n])

    return FourierTrans_ft

# Version 4: Modularized
def FilterFunc(freqs, timePulses, maxTime):
    """
    PARAMETERS:
    freqs (np.ndarray, 1D): frequencies in Hertz. Must be ordered from least to greatest
    timePulses (np.ndarray, 1D): times at which pulses are applied, sortted from lest to greatest
    tol (float): used to determine when to use Taylor expansion of Filter function around angular freq * time = 0
    RETURNS:
    FF (np.ndarray, 1D): filter function for each frequency
    """
    FourierTrans_ft = fTilde(freqs, timePulses, maxTime) 
    FF = ( FourierTrans_ft * FourierTrans_ft.conj() ).real # take real to convert array to real data type
    
    return FF

# Version 3: Sinc
#def FilterFunc(freqs, timePulses, maxTime, tol=0.1):
#    """
#    PARAMETERS:
#    freqs (np.ndarray, 1D): frequencies in Hertz. Must be ordered from least to greatest
#    timePulses (np.ndarray, 1D): times at which pulses are applied, sortted from lest to greatest
#    tol (float): used to determine when to use Taylor expansion of Filter function around angular freq * time = 0
#    RETURNS:
#    FF (np.ndarray, 1D): filter function for each frequency
#    """
#    Np = len(timePulses)
#    Nf = len(freqs)
#    
#    timeWithEnds = np.zeros(Np+2)
#    timeWithEnds[0] = 0
#    timeWithEnds[-1] = maxTime
#    timeWithEnds[1:Np+1] = timePulses
#
#    a = timeWithEnds[1:] - timeWithEnds[:Np+1]
#    b = (timeWithEnds[1:] + timeWithEnds[:Np+1]) / 2
#    
#    FourierTrans_ft = np.zeros(Nf, dtype='complex')
#    FF = np.zeros(Nf)
#    
#    # Make more efficient using vectorization?
#    for n in range(Np+1):
#        FourierTrans_ft += (-1)**n * np.exp(-1j * 2 * np.pi * freqs * b[n]) * a[n] * np.sinc(freqs * a[n])
#
#    FF = ( FourierTrans_ft * FourierTrans_ft.conj() ).real # take real to convert array to real data type
#    
#    return FF

# VERSION2: Full exponentials
#def FilterFunc(freqs, timePulses, maxTime, tol=0.1):
#    """
#    PARAMETERS:
#    freqs (np.ndarray, 1D): frequencies in Hertz. Must be ordered from least to greatest
#    timePulses (np.ndarray, 1D): times at which pulses are applied
#    tol (float): used to determine when to use Taylor expansion of Filter function around angular freq * time = 0
#    RETURNS:
#    filterFunc (np.ndarray, 1D): filter function for each frequency
#    """
#    Np = len(timePulses)
#    Nf = len(freqs)
#    
#    timeWithEnds = np.zeros(Np+2)
#    timeWithEnds[0] = 0
#    timeWithEnds[-1] = maxTime
#    timeWithEnds[1:Np+1] = timePulses
#    
#    alphas = (-1)**(np.arange(Np+2) + 1) # array that looks like [-1, 1, -1, 1, -1, ... , (-1)**(Np+1)]
#    alphas[1:Np+1] *= 2 # array now looks like [-1, 2, -2, 2, -2, ... , (-1)**(Np+1)]
#    
#    angFreqs = 2 * np.pi * freqs
#
#    tol_idx = 0 # Index where angFreqs * maxTime < tol
#    for i in range(Nf):
#        if angFreqs[i] * maxTime > tol:
#            tol_idx = i
#            break # Exit for loop
#
#    afMesh, tMesh = np.meshgrid(angFreqs, timeWithEnds, sparse=True)
#    phasor = np.zeros((Np+2, Nf), dtype='complex')
#
#    # NOTE: afMesh.shape = (1, Nf) and tMesh.shape = (Np+2, 1).
#    # This means (afMesh * tMesh).shape = (Np+2, Nf).
#   
#    # Use power series expansion of the phasor coefficients when angFreqs * maxTime < tol:
#    # Get frequencies to the left of the tolerance cutoff.
#    afMeshLeft = afMesh[:,:tol_idx]  
#    phasor[:, :tol_idx] = tMesh - 1j * afMeshLeft * tMesh**2 / 2 - afMeshLeft**2 * tMesh**3 / 6 + 1j * afMeshLeft**3 * tMesh**4 / 24 + afMeshLeft**4 * tMesh**5 / 120
#
#    # Use full expression for rest of phasor coefficients
#    afMeshRight = afMesh[:, tol_idx:Nf]
#    phasor[:, tol_idx:Nf] = np.exp(-1j * afMeshRight * tMesh) / afMeshRight
#
#    FT_ft = np.matmul(alphas, phasor) # Fourier Transform of f(t). Shape (freqs)
#    filterFunc = (FT_ft * FT_ft.conj()).real # take real to convert array to real data type
#    
#    return filterFunc

# VERSION 1
#def FilterFunc(freqs, timePulses, maxTime):
#    """
#    PARAMETERS:
#    freqs (np.ndarray, 1D): frequencies in Hertz
#    timePulses (np.ndarray, 1D): times at which pulses are applied
#    RETURNS:
#    filterFunc (np.ndarray, 1D): filter function for each frequency
#    """
#    Np = len(timePulses)
#    Nf = len(freqs)
#    signs = (-1)**np.arange(Np+1) # array that looks like [1, -1, 1, -1, ...]
#    timeWithEnds = np.zeros(Np+2)
#    timeWithEnds[0] = 0
#    timeWithEnds[-1] = maxTime
#    timeWithEnds[1:Np+1] = timePulses
#
#    phasor = np.zeros((Np+1, Nf), dtype='complex')
#    angFreqs = 2 * np.pi * freqs
#    for i in range(Np+1):
#        t1 = timeWithEnds[i]
#        t2 = timeWithEnds[i+1]
#        numerator = np.exp( - 1j* angFreqs * t2) - np.exp( -1j * angFreqs * t1)
#        # If complex arg is small enough, just do linear order Taylor expansion of terms in the sum
#        phasor[i] = np.divide(numerator, angFreqs, out = np.full(Nf, t2 - t1, dtype='complex'), where = angFreqs != 0)
#    
#    FT_ft = np.matmul(signs, phasor) # Fourier Transform of f(t). Shape (freqs)
#    filterFunc = (FT_ft * FT_ft.conj()).real # take real to convert array to real data type
#    
#    return filterFunc

def randomSequence(nPulseChances):
    return np.random.randint(low=0, high=2, size=nPulseChances)

def pulse_times(pulse_seq, max_time):
    "state is now pulse_seq"
    Nc = len(pulse_seq)
    
    tau = max_time / (Nc + 1)
    
    time = tau * np.arange(1, Nc + 1) # Times at which agent allowed to apply a pulse
    time_applied = time[pulse_seq > 0] # Times at which a pulse is actually applied

    return time_applied

# Version 2
def timeAgentApply(state, maxTime):
    nPulseChances = len(state)
    fullSign = np.ones(nPulseChances + 1) # Full sign is just state with extra 1 at the front
    fullSign[1:] = state
    
    agentSequence = np.zeros(nPulseChances, dtype=int)
    for i in range(nPulseChances):
        if fullSign[i] * fullSign[i+1] < 0:
            agentSequence[i] = 1
    
    tau = maxTime / (nPulseChances + 1)
    
    time = tau * np.arange(1, nPulseChances + 1) # Times at which agent allowed to apply a pulse
    TAA = time[agentSequence > 0] # Times at which the agent chose to apply a pulse

    return TAA

# Version 1:
#def timeAgentApply(nPulseChances, state, maxTime):
#    fullSign = np.ones(nPulseChances + 1) # Full sign is just state with extra 1 at the front
#    fullSign[1:] = state
#    
#    agentSequence = np.zeros(nPulseChances, dtype=int)
#    for i in range(nPulseChances):
#        if fullSign[i] * fullSign[i+1] < 0:
#            agentSequence[i] = 1
#    
#    tau = maxTime / (nPulseChances + 1)
#    
#    time = tau * np.arange(1, nPulseChances + 1) # Times at which agent allowed to apply a pulse
#    TAA = time[agentSequence > 0] # Times at which the agent chose to apply a pulse
#
#    return TAA

def make_freq(maxTime, nPulseChances, nFreq = 500):
    tau = maxTime / (nPulseChances + 1)
    max_freq = 1 / 2 / tau
    freq_mesh = np.linspace(0, max_freq, nFreq)
    return freq_mesh

#def make_noise(freqs, mu_frac, cutoff_frac, cutoff_noise = 1e-6):
#    mu = mu_frac * freqs[-1]
#    cutoff_freq = cutoff_frac * freqs[-1]
#    temperature = (cutoff_freq - mu) / np.log(1 / cutoff_noise - 1)
#    return 1 / (np.exp( (freqs - mu) / temperature ) + 1)

def temp_from_cutoff(chem_potential, cutoff_freq, cutoff_noise = 1e-6):
    return (cutoff_freq - chem_potential) / np.log( 1/cutoff_noise - 1 )

def fermi_dirac(freqs, chem_potential, temperature):
    return 1 / (np.exp( (freqs - chem_potential) / temperature ) + 1)

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
    if weights is None:
        dfreq = freqs[1] - freqs[0]
        return np.sum(filter_func * noise * dfreq)
    else:
        return np.sum(filter_func * noise * weights)

def chi_avg(freqs, noise, maxTime, nPulseChances, weights=None):
    pulse_spacing = maxTime / (nPulseChances + 1)
    ff = (nPulseChances + 1) * (pulse_spacing * np.sinc(freqs * pulse_spacing))**2
    return chi(freqs, noise, ff, weights=weights)

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
    return initial_chi / chi_array 

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

if __name__=='__main__':
    
    # Load data

    param = np.loadtxt('../data/param.txt')
    tMax = param[0]
    nPulseTot = int(param[1]) # Total pulses agent allowed to apply
    nFreq = int(param[2])
    chemPotential = param[-2]
    temperature = param[-1]

    freq = np.loadtxt('../data/freq.txt')
    sModSq = np.loadtxt('../data/sOmegaAbs2.txt')
    agentFilter = np.loadtxt('../data/fOmegaAbs2.txt')
    
    finalState = np.loadtxt('../data/state.txt')
    reward = np.loadtxt('../data/reward.txt')
    loss = np.loadtxt('../data/loss.txt')
    
    # Crunch numbers

    tau = tMax / (nPulseTot + 1)

    agentTime = timeAgentApply(nPulseTot, finalState, tMax) # Times at which the agent chose to apply a pulse
    nPulseApp = len(agentTime) # Number of pulses the agent actually applied
   
    UDDTime = UDD(nPulseApp, tMax)
    CPMGTime = CPMG(nPulseApp, tMax)

    #agentFilter = FilterFunc(freq, agentTime, tMax)
    UDDFilter = FilterFunc(freq, UDDTime, tMax)
    CPMGFilter = FilterFunc(freq, CPMGTime, tMax)

    UDDReward = RewardFunc(freq, sModSq, UDDFilter, tau, nPulseTot)
    CPMGReward = RewardFunc(freq, sModSq, CPMGFilter, tau, nPulseTot)

    dfreq = freq[1] - freq[0]
    agentOverlap = np.sum(agentFilter * sModSq * dfreq)
    UDDOverlap = np.sum(UDDFilter * sModSq * dfreq)
    CPMGOverlap = np.sum(CPMGFilter * sModSq * dfreq)

    nTrial = len(reward)

    # Plot

    fig = plt.figure(layout = 'constrained', figsize = (16, 8))
    mosaic = """
             ADF
             BDG
             CE.
             """
    axd = fig.subplot_mosaic(mosaic, width_ratios=[1, 2, 2])
    
    UDDColor = 'blue'
    UDDLinestyle = 'solid'
    UDDLabel = r'$\chi_{UDD} = $' + '{:.3e}'.format(UDDOverlap)

    agentColor = 'purple'
    agentLabel = r'$\chi_{Agent} = $' + '{:.3e}'.format(agentOverlap)
    
    CPMGColor = 'red'
    CPMGLinestyle = 'solid'
    CPMGLabel = r'$\chi_{CPMG} = $' + '{:.3e}'.format(CPMGOverlap)

    lwps = 2 # Linewidth for pulse sequence plots
    
    # Plot pulse sequence over time

    axd['A'].set_title('UDD ($N_{{pulse}} = {}$)'.format(nPulseApp))
    axd['A'].sharex(axd['C'])
    axd['A'].vlines(UDDTime, 0, 1, color=UDDColor, linestyle=UDDLinestyle, linewidth = lwps)
    axd['A'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['A'].tick_params(labelbottom=False) # Remove x axis labels
   
    axd['B'].set_title('Agent ($N_{{allowed}} = {0}, N_{{pulse}} = {1}$)'.format(nPulseTot, nPulseApp))
    axd['B'].sharex(axd['C'])
    axd['B'].vlines(agentTime, 0, 1, color=agentColor, linewidth = lwps)
    axd['B'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['B'].tick_params(labelbottom=False) # Remove x axis labels
   
    axd['C'].set_title('CPMG ($N_{{pulse}} = {}$)'.format(nPulseApp))
    axd['C'].set_xlabel('Time')
    axd['C'].set_xlim(0, tMax)
    axd['C'].vlines(CPMGTime, 0, 1, color=CPMGColor, linestyle=CPMGLinestyle, linewidth=lwps)
    axd['C'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks

    # Plot filter functions and noise PSD
   
    axd['D'].sharex(axd['E'])
    axd['D'].plot(freq, UDDFilter, color=UDDColor, linestyle=UDDLinestyle, label = UDDLabel)
    axd['D'].plot(freq, agentFilter, color=agentColor, label = agentLabel)
    axd['D'].plot(freq, CPMGFilter, color=CPMGColor, linestyle=CPMGLinestyle, label = CPMGLabel)
    axd['D'].set_ylabel(r'$F(\nu)$')
    axd['D'].tick_params(labelbottom=False) # Remove x axis labels
    axd['D'].legend()

    axd['E'].set_ylabel(r'$|S(\nu)|^2$')
    axd['E'].set_xlabel(r'$\nu$ [1/time]')
    axd['E'].set_title('$\mu = {0}, T = {1}$'.format(chemPotential, temperature))
    axd['E'].plot(freq, sModSq)

    # Plot reward and loss over trials
    
    axd['F'].set_ylabel('Reward')
    axd['F'].sharex(axd['G'])
    axd['F'].plot(range(1, nTrial + 1), reward) # Do range(1, n+1) so that index starts at 1

    axd['G'].set_ylabel('Average Loss')
    axd['G'].set_xlabel('Trials')
    axd['G'].plot(range(1, nTrial + 1), loss)

    #fig.tight_layout()
    plt.show()
