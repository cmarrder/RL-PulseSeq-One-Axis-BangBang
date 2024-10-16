import numpy as np
import matplotlib.pyplot as plt
import pulse_sequence as ps
import os

plotCurves = True

# Load data

oDir = '/home/charlie/Documents/ml/CollectiveAction/data/job_00000'

nPulse = int( np.loadtxt(os.path.join(oDir, 'nPulse.txt')) ) # Number of pulse applications
#nTimeStep = int( np.loadtxt(os.path.join(oDir, 'nTimeStep.txt')) )# Number of pulse chances/locations
tMax = np.loadtxt(os.path.join(oDir, 'maxTime.txt'))
#chemPotential = np.loadtxt(os.path.join(oDir, 'noiseParam1.txt'))
#temperature = np.loadtxt(os.path.join(oDir, 'noiseParam2.txt'))

freq = np.loadtxt(os.path.join(oDir, 'freq.txt'))
#sOmega = np.loadtxt(os.path.join(oDir, 'sOmega.txt'))

#finalState = np.loadtxt(os.path.join(oDir, 'state.txt'))
#reward = np.loadtxt(os.path.join(oDir, 'reward.txt'))
#loss = np.loadtxt(os.path.join(oDir, 'loss.txt'))

# Choose noise PSD profile

cpmgPeak = nPulse / 2 / tMax
chemPotential = cpmgPeak / 8
temperature = 0.4#0.01
# Fermi-Dirac distribution for studying hard to smooth cutoff (low temp to high temp)
S = ps.fermi_dirac(freq, chemPotential, temperature)

# Sum of Lorentzians peaked at the CPMG filter function peaks so we make it hard for CPMG to perform well
#tau = tMax / (2 * nPulse)
#centers = (np.arange(nPulse) + 0.5) / (2 * tau)
#fwhms = np.ones_like(centers) / 2
#S = ps.lorentzians(freq, centers, fwhms)
## Normalize if you wish:
#S /= len(centers)

# Crunch numbers
UDDTime = ps.UDD(nPulse, tMax)
PDDTime = ps.PDD(nPulse, tMax)
CPMGTime = ps.CPMG(nPulse, tMax)
print("\nUDD")
print(UDDTime)
print("\nPDD")
print(PDDTime)
print("\nCPMG")
print(CPMGTime)

#UDDSteps = np.diff(UDDTime)
#CPMGSteps = np.diff(CPMGTime)

if plotCurves == True:

    UDDFilter = ps.FilterFunc(freq, UDDTime, tMax)
    CPMGFilter = ps.FilterFunc(freq, CPMGTime, tMax)
    
    dfreq = freq[1] - freq[0]
    UDDOverlap = np.sum(S * UDDFilter * dfreq)
    CPMGOverlap = np.sum(S * CPMGFilter * dfreq)
    
    UDDColor = 'blue'
    CPMGColor = 'red'
    
    fig, axs = plt.subplots(3, sharex=True, layout='constrained')
    
    string_Na = '$N_{{pulse}} = {}$, '.format(nPulse)
    string_mu = '$\mu$ = {:.3e}, '.format(chemPotential)
    string_temp = '$T$ = {:.3e}'.format(temperature)
    
    # Find max index for plotting
    for i in range(len(S)-1, -1, -1):
        if (S[i] / freq[i] < 1e-12):
            maxIdx = i
    
    #axs[0].set_title('$N_{{pulse}} = {}$'.format(nPulse))
    axs[0].set_title(string_Na + string_mu + string_temp)
    axs[0].plot(freq[:maxIdx], S[:maxIdx])
    axs[0].set_ylabel(r"$S(\nu)$")
    
    axs[1].plot(freq[:maxIdx], UDDFilter[:maxIdx], label = "UDD", color = UDDColor)
    axs[1].plot(freq[:maxIdx], CPMGFilter[:maxIdx], label = "CPMG", color = CPMGColor)
    axs[1].set_ylabel(r"$F(\nu)$")
    axs[1].legend()
    
    axs[2].plot(freq[:maxIdx], S[:maxIdx]*UDDFilter[:maxIdx], label = '$\chi_{{UDD}}$ = {:.4e}'.format(UDDOverlap), color = UDDColor)
    axs[2].plot(freq[:maxIdx], S[:maxIdx]*CPMGFilter[:maxIdx], label = '$\chi_{{CPMG}}$ = {:.4e}'.format(CPMGOverlap), color = CPMGColor)
    axs[2].set_ylabel(r"$F(\nu) * S(\nu)$")
    axs[2].set_xlabel(r"$\nu$ in Hz")
    axs[2].legend()
    
    plt.show()
