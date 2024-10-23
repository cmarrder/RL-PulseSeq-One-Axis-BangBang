import numpy as np
import matplotlib.pyplot as plt
import pulse_sequence as ps
import os

def FilterFunc2(freqs, pulse_times, max_time):
    """ Calculated the filter function in the same way ps.FilterFunc does,
        but using an expression derived from the one ps.FilterFunc uses

        Note: gives issue at points where freqs close to or equal to zero. Divide error.
    """
    Np = len(pulse_times)
    Nfreq = len(freqs)

    cmaxtime = np.cos( 2 * np.pi * freqs * max_time)
    smaxtime = np.sin( 2 * np.pi * freqs * max_time)

    alphas = 2 * (-1)**( np.arange(1, Np+1) + 1)
    
    cosines = np.zeros((Np, Nfreq))
    sines = np.zeros((Np, Nfreq))
    for i in range(Np):
        cosines[i] = np.cos(2 * np.pi * freqs * pulse_times[i])
        sines[i] = np.sin(2 * np.pi * freqs * pulse_times[i])
    cweighted = np.matmul(alphas, cosines) # shape (Nfreq,)?
    sweighted = np.matmul(alphas, sines)

    double_sum = np.zeros(Nfreq)
    for n in range(0, Np-1):
        for m in range(n+1, Np):
            double_sum += alphas[n] * alphas[m] * (cosines[n] * cosines[m] + sines[n] * sines[m])


    term1 = 2 + 2 * (-1)**(Np + 1) * cmaxtime 
    term2 = 2 * (-1 + (-1)**Np * cmaxtime) * cweighted
    term3 = 2 * (-1)**Np * smaxtime * sweighted
    term4 = 4 * Np + 2 * double_sum

    return (term1 + term2 + term3 + term4) / (2 * np.pi * freqs)**2

def approxFilterFunc(freqs, sign_seq, max_time):
    """ Calculates the filter function associated with some f(t)
        (which is represented by sign_seq) by discretizing time.
    """
    Ndeltas = len(sign_seq)
    dt = max_time / Ndeltas

    prefactor = dt**2 * np.sinc(freqs * dt)**2
    times = (np.arange(Ndeltas) + 0.5) * dt

    freq_mesh, time_mesh = np.meshgrid(freqs, times, sparse=True)
    phasor = np.exp( - 1j * 2 * np.pi * freq_mesh * time_mesh ) # shape (Ndeltas, Nfreqs)
    alpha = np.matmul(sign_seq, phasor)
    mod_sq_alpha = (alpha * alpha.conj()).real
    return prefactor * mod_sq_alpha
    #return prefactor
    #return mod_sq_alpha

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

def CPMGPulseSeq(Npa, idx_step):
    """
    Returns an array which represents a discretized form of CPMG,
    with zeros representing no pulse applied at that index,
    and ones representing a pulse applied.
    Npa (int): number of pulse applications
    idx_step (int): size of the time steps between pulses in terms of indices
    """
    Npc = 2 * Npa * idx_step - 1

    pulse_seq_ends = np.zeros(Npc + 2) # Pulse sequence with added endpoints for initial and final times

    idx = idx_step
    for i in range(Npa):
        pulse_seq_ends[idx] = 1
        idx += 2 * idx_step

    pulse_seq = pulse_seq_ends[1 : Npc + 1] # Chop off endpoints

    return pulse_seq

def timeBetween(Nb, Nc, max_time):
    """ Function to calculate the time between pulse chances.
    """
    Nt = (Nc + 1) * Nb
    time = np.zeros(Nt)
    dt = max_time / Nt

    for j in range(Nt):
        time[j] = dt * (0.5 + j)

    return time

def timeChances(Nc, max_time):
    """ Function to calculate the times of pulse chances.
    """
    time = np.zeros(Nc)
    tau = max_time / (Nc + 1)
    
    for k in range(Nc):
        time[k] = (k + 1) * tau

    return time

def testTimeMesh(Nc, Nb):
    # Nc (int): Number of pulse chances
    # Nb (int): Number of points between pulse chances
    Nt = (Nc + 1) * Nb
    maxTime = Nt
    dt = maxTime / Nt

    tBetween = timeBetween(Nb, Nc, maxTime)
    tChances = timeChances(Nc, maxTime)

    print('tBetween')
    print(tBetween)
    print('tChances')
    print(tChances)

    plt.scatter(tBetween, np.ones_like(tBetween), label = 'tBetween')
    plt.scatter(tChances, np.zeros_like(tChances), label = 'tChances')
    plt.legend()
    plt.show()

    return

def testFreq(Nc, Nb, max_time):
    # Nc (int): Number of pulse chances
    # Nb (int): Number of points between pulse chances
    Nt = (Nc + 1) * Nb
    dt = max_time / Nt

    J = int( (Nt - 1 - (Nt - 1) % 2) / 2 )
    L = int( - (Nt - Nt % 2) / 2 )


    print('Nt = {}'.format(Nt))
    print('Nc = {}'.format(Nc))
    print('Nb = {}'.format(Nb))
    print('J = {}'.format(J))
    print('L = {}'.format(L))
    r = np.zeros(Nt)
    r[0 : J+1] = np.linspace(0, J, num = int(J+1))
    r[J+1 : Nt] = np.linspace(L, -1, num = -L)
    nu = r / (Nt * dt)
    
    freq = np.fft.fftfreq(Nt, d=dt)

    print('J = {}'.format(J))
    print('r')
    print(r)
    print('nu')
    print(nu)
    print('fftfreq')
    print(freq)
    return

def byte2complex(byte):
    # str(byte) looks like:
    # b'(4,0)'

    # str(byte)[3:-2] looks like:
    # 4,0

    # str(byte)[3:-2].replace("," , "+") + "j" looks like:
    # 4+0j
    string = str(byte)[3:-2]

    # If the imaginary part is negative, replace ,- with -
    # If the imaginary part is positive, then nothing is done.
    modified_string = string.replace(",-", "-")
    # If the imaginary part is positive, replace , with +
    modified_string = modified_string.replace(",", "+")

    return complex(modified_string + "j")

#def locpeaksFT(pulseTimes, nLocations):
#    """ Peaks of sinc function
#    """
#    #Na = len(pulseTimes)
#    an = np.diff(pulseTimes)
#    #q = np.arange(Na) + 0.5
#    q = np.arange(int((nLocations + 1) / 2)) + 0.5
#    rhs = q - 1/q - 2/(3*q**3) - 13/(15*q**5) - 146/(105*q**7)
#    locations = np.unique( np.outer(q, 1/an).flatten() )
#    return locations

def locpeaksFT(pulse_times, max_time, freqs):
    """ Peaks of fourier transform of f(t).
        pulse_times and freqs must be sorted.
        THIS DOES NOT WORK CURRENTLY
    """
    nPulse = len(pulse_times)
    times_with_ends = np.zeros(nPulse + 2)
    times_with_ends[1:nPulse+1] = pulse_times
    times_with_ends[-1] = max_time
    an = np.diff(times_with_ends)

    max_an = np.max(an)
    max_freq = freqs[-1]
    max_j = int(max_freq * max_an - 0.5)

    q = (np.arange(max_j + 1) + 0.5) * np.pi # Plus one in arange to include max_j value
    rhs = q - 1/q - 2/(3*q**3) - 13/(15*q**5) - 146/(105*q**7)
    peak_freqs = np.unique( np.outer(rhs, 1/an).flatten() ) / np.pi
    return peak_freqs

def locpeaksCPMG(nPulse, nLocation, max_time, freqs):
    tau = max_time / (2 * nPulse) 
    print("tau = {}".format(tau))
    print("1 / tau = {}".format(1/tau))
    print("1 / (tau/2) = {}".format(2/tau))
    print("1 / (2 tau) = {}".format(1/2/tau))
    print("1 / (4 tau) = {}".format(1/4/tau))
    #k = 0
    #peak_freq_list = []
    #q = (k + 0.5) * np.pi
    #expansion = q - q**(-1) - 2/3 * q**(-3) - 13/15 * q**(-5) - 146/105 * q**(-7)
    #peak_freq = expansion / (2 * np.pi * tau)
    #while peak_freq < freqs[-1]:
    #    peak_freq_list.append(peak_freq)
    #    k += 1
    #    q = (k + 0.5) * np.pi
    #    expansion = q - q**(-1) - 2/3 * q**(-3) - 13/15 * q**(-5) - 146/105 * q**(-7)
    #    peak_freq = expansion / (2 * np.pi * tau)
    #
    #return np.array(peak_freq_list)

    peak_freqs = (np.arange(2 * nPulse) + 0.5) / (2 * tau)
    return peak_freqs


#def fig_FT_comparison(max_time, freq, noise, sign_seq, pulse_times, scale='linear'):
def fig_FT_comparison(max_time,
              approxFreq, approxNoise, approxFT,
              exactFreq, exactNoise, exactFT,
              scale='linear', vertical_lines=None):

    approxFilter = (approxFT * approxFT.conj()).real
    exactFilter = (exactFT * exactFT.conj()).real

    # Calculate overlap
    approxDfreq = approxFreq[1] - approxFreq[0]
    approxOverlap = np.sum(approxNoise * approxFilter * approxDfreq)
    
    exactDfreq = exactFreq[1] - exactFreq[0]
    exactOverlap = np.sum(exactNoise * exactFilter * exactDfreq)

    # Calculate reward
    exactReward = ps.RewardFunc(exactOverlap)#1 / exactOverlap
    approxReward = ps.RewardFunc(approxOverlap)#1 / approxOverlap
   
    print('\nExact Overlap')
    print(exactOverlap)
    print('FFT Overlap')
    print(approxOverlap)
    
    print('\nExact Reward')
    print(exactReward)
    print('FFT Reward')
    print(approxReward)

    fig = plt.figure(layout = 'constrained', figsize = (14, 7))
    mosaic = """
             A.
             CD
             EF
             """
    axd = fig.subplot_mosaic(mosaic)
    
    # Plot real part of Fourier transform of F
    axd['A'].plot(approxFreq, approxFT.real, label = "Approx")
    axd['A'].plot(exactFreq, exactFT.real, label = "Exact", linestyle = 'dashed')
    axd['A'].set_ylabel(r"$Re[\tilde{f}(\nu)]$")
    axd['A'].legend()
    axd['A'].set_yscale(scale)
    axd['A'].sharex(axd['E'])
    
    # Plot imaginary part of Fourier transform of F
    axd['C'].plot(approxFreq, approxFT.imag, label = "Approx")
    axd['C'].plot(exactFreq, exactFT.imag, label = "Exact", linestyle = 'dashed')
    axd['C'].set_ylabel(r"$Im[\tilde{f}(\nu)]$")
    axd['C'].legend()
    axd['C'].set_yscale(scale)
    axd['C'].sharex(axd['E'])
    
    # Plot filters
    axd['E'].plot(approxFreq, approxFilter, label = "Approx")
    axd['E'].plot(exactFreq, exactFilter, label = "Exact", linestyle = 'dashed')
    axd['E'].set_ylabel(r"$F(\nu)$")
    axd['E'].legend()
    axd['E'].set_yscale(scale)
    axd['E'].set_xlabel(r"$\nu$ in Hz")
    if vertical_lines is None:
        pass
    else:
        axd['E'].vlines(vertical_lines, 0, np.max(approxFilter), colors='red')
    
    # Plot of noise
    axd['D'].plot(approxFreq, approxNoise, label = "Approx") 
    axd['D'].plot(exactFreq, exactNoise, label = "Exact", linestyle = 'dashed')
    axd['D'].legend()
    axd['D'].set_ylabel(r"$|S(\nu)|^2$")
    axd['D'].set_yscale(scale)
    axd['D'].sharex(axd['F'])
    
    # Plot noise times filters
    axd['F'].plot(approxFreq, approxNoise*approxFilter, label = '$\chi_{{approx}}$ = {:.4e}'.format(approxOverlap))
    axd['F'].plot(exactFreq, exactNoise*exactFilter, label = '$\chi_{{exact}}$ = {:.4e}'.format(exactOverlap), linestyle = 'dashed')
    axd['F'].set_ylabel(r"$F(\nu) * S|(\nu)|^2$")
    axd['F'].legend()
    axd['F'].set_yscale(scale)
    axd['F'].set_xlabel(r"$\nu$ in Hz")

    return fig

def compareFT(nPulse,
              nLocation,
              max_time,
              nBetween,
              nTimePts,
              chem_pot,
              temperature,
              scale='linear', test_CPMG=False, read_dir=None, plot_peaks=False):
    cpmgStep = int( (nLocation + 1) / (2 * nPulse) )
    nSample = (nLocation + 1) * nBetween
    dt = max_time / nSample

    nZero = nTimePts - nSample
    if read_dir is None:
        J = int( (nTimePts - 1 - (nTimePts - 1) % 2) / 2 )
        # Calculate frequencies
        approxFreq = np.fft.fftfreq(nTimePts, d=dt)[:J+1] # Select only positive frequencies
        approxNoise = ps.fermi_dirac(approxFreq, chem_pot, temperature)
        # Calculate pulse sequence and sign sequence
        if test_CPMG:
            pulse_seq = CPMGPulseSeq(nPulse, cpmgStep)
        else:
            pulse_seq = ps.randomSequence(nLocation)
        sign_seq = calc_sign_seq(pulse_seq, Nb=nBetween)
        padded_sign_seq = np.concatenate((sign_seq, np.zeros(nZero)))
        # Calculate approximate Filter function.
        # Select only positive frequencies. Technically don't need global phase, just accounting because we can.
        globalPhase = np.exp(-1j*2*np.pi*dt/2 * approxFreq) # In our analytical solution, we don't take time to start at 0, but rather at dt/2. 
        # Calculate fft. Select only positive frequencies. Normalize.
        approxFtilde = np.fft.fft(padded_sign_seq, norm="backward")[:J+1] * globalPhase 
        approxFtilde *= dt 
        
        # Calculate exact pulse times
        if test_CPMG:
            exactTime = ps.CPMG(nPulse, max_time)
        else:
            exactTime = ps.pulse_times(pulse_seq, max_time)
    else:
        freq_file = os.path.join(read_dir, "freq.txt")
        fourier_file = os.path.join(read_dir, "fOmega.txt")
        noise_file = os.path.join(read_dir, "sOmegaAbs2.txt")
        times_file = os.path.join(read_dir, "state.txt")

        approxFreq = np.loadtxt(freq_file)
        approxFtilde = np.loadtxt(fourier_file, dtype = 'complex', converters = byte2complex)
        approxNoise = np.loadtxt(noise_file)
        exactTime = np.loadtxt(times_file)
   
        # Do some normalizing:
        globalPhase = np.exp(-1j*2*np.pi*dt/2 * approxFreq) # In our analytical solution, we don't take time to start at 0, but rather at dt/2. 
        #approxFtilde *= dt * globalPhase
        approxFtilde *= globalPhase

    # Calculate exact Fourier transform
    exactFreq = np.linspace(0, approxFreq[-1], 10*nTimePts)
    exactFtilde = ps.fTilde(exactFreq, exactTime, max_time)
    exactNoise = ps.fermi_dirac(exactFreq, chem_pot, temperature)
        
    
    # Decide whether to plot vertical lines at main peaks or not
    if plot_peaks == True:
        if test_CPMG == True:
            freq_peaks = locpeaksCPMG(nPulse, nLocation, max_time, exactFreq)
        else:
            freq_peaks = locpeaksFT(exactTime, max_time, exactFreq)
    else:
        freq_peaks = None
    
    print("Time of Pulses:")
    print(exactTime)

    fig = fig_FT_comparison(max_time, approxFreq, approxNoise, approxFtilde, exactFreq, exactNoise, exactFtilde, scale=scale, vertical_lines=freq_peaks)
    plt.suptitle('$N_{{pulse}} = {}$'.format(nPulse))
    plt.show()
    return

if __name__=='__main__':
    
    print('')

    read_output = False
    read_path = "/home/charlie/Documents/ml/Filter/job_test" 

    scale = 'linear'#'log'
    test_CPMG = True
    plot_peaks = True

    if read_output == True:
        read_directory = read_path
        init_file = os.path.join(read_path, "initials.txt")
        init_array = np.loadtxt(init_file)
        nPulseApp = int(init_array[0])
        nPulseChance = int(init_array[1])
        tMax = init_array[2]
        nB = int(init_array[3])
        nT = int(init_array[4])
        chem_potential = init_array[5]
        temp = init_array[6]
    else:
        read_directory = None
        nPulseApp = 8
        int_step = 7#3#2
        nPulseChance = 2 * int_step * nPulseApp - 1
        tMax = 1#3#10
        nB = 10 # Number of time points between pulses
        nS = (nPulseChance + 1) * nB # Number of time signal samples before the zero padding
        nZ = 20*nS#10000 # Number of zeros padded to the end
        nT = nS + nZ
        chem_potential = 0.1 
        temp = 0.1

    J = int( (nT - 1 - (nT - 1) % 2) / 2 )
    print('Nc: {}'.format(nPulseChance))
    print('Nt: {}'.format(nT))
    print('J: {}'.format(J))
    print('tMax: {}'.format(tMax))
    compareFT(nPulseApp, nPulseChance, tMax, nB, nT, chem_potential, temp, scale=scale, test_CPMG=test_CPMG, read_dir=read_directory, plot_peaks=plot_peaks)

