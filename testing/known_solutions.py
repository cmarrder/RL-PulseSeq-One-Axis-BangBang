import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pulse_sequence as ps
import os

plotCurves = True

# Load data

oDir = '/home/charlie/Documents/ml/CollectiveAction/eta_scan_data/1_over_f/harmonics_01_02_08/job_00002/run_00001'

nPulse = int( np.loadtxt(os.path.join(oDir, 'nPulse.txt')) ) # Number of pulse applications
#nTimeStep = int( np.loadtxt(os.path.join(oDir, 'nTimeStep.txt')) )# Number of pulse chances/locations
tMax = np.loadtxt(os.path.join(oDir, 'maxTime.txt'))
param1 = np.loadtxt(os.path.join(oDir, 'noiseParam1.txt'))
param2 = np.loadtxt(os.path.join(oDir, 'noiseParam2.txt'))

freq = np.loadtxt(os.path.join(oDir, 'freq.txt'))
sOmega = np.loadtxt(os.path.join(oDir, 'sOmega.txt'))

#finalState = np.loadtxt(os.path.join(oDir, 'state.txt'))
#reward = np.loadtxt(os.path.join(oDir, 'reward.txt'))
#loss = np.loadtxt(os.path.join(oDir, 'loss.txt'))

# Choose noise PSD profile

cpmgPeak = nPulse / 2 / tMax
param1 = cpmgPeak / 8
param2 = 0.4#0.01
# Fermi-Dirac distribution for studying hard to smooth cutoff (low temp to high temp)
#S = ps.fermi_dirac(freq, param1, param2)

S = sOmega
noise = sOmega

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
    PDDFilter = ps.FilterFunc(freq, PDDTime, tMax)
    CPMGFilter = ps.FilterFunc(freq, CPMGTime, tMax)
    
    UDDOverlap = ps.chi(freq, S, UDDFilter)
    PDDOverlap = ps.chi(freq, S, PDDFilter)
    CPMGOverlap = ps.chi(freq, S, CPMGFilter)

    UDDFid = ps.fidelity(UDDOverlap)
    PDDFid = ps.fidelity(PDDOverlap)
    CPMGFid = ps.fidelity(CPMGOverlap)

    UDDLabel = '$p_{{UDD}} = {:.3f}$'.format(UDDFid)#'$\chi_{{UDD}}$ = {:.4e}'.format(UDDOverlap)
    PDDLabel = '$p_{{PDD}} = {:.3f}$'.format(UDDFid)#'$\chi_{{UDD}}$ = {:.4e}'.format(UDDOverlap)
    CPMGLabel = '$p_{{CPMG}} = {:.3f}$'.format(CPMGFid)#'$\chi_{{CPMG}}$ = {:.4e}'.format(CPMGOverlap)
    
    string_Na = '$N_{{pulse}} = {}$, '.format(nPulse)
    string_mu = '$\mu$ = {:.3e}, '.format(param1)
    string_temp = '$T$ = {:.3e}'.format(param2)
    #title = string_Na + string_mu + string_temp
    title = 'Performance of UDD, PDD, and CPMG on $1/f$ Noise'
    
    # Find max index for plotting
    cutoffIdx = len(S)-1 # Initialize
    for i in range(len(S)-1, -1, -1):
        if (S[i] / freq[i] > 1e-3):
            cutoffIdx = i
            break

    fig = plt.figure(layout = 'constrained', figsize = (16, 8))
    mosaic = """
             AD
             BE
             CF
             """
    axd = fig.subplot_mosaic(mosaic, width_ratios=[1, 2])
    
    UDDColor = '#377EB8'# Blue
    UDDLinestyle = 'solid'

    PDDColor = '#984EA3'# Lilac
    PDDLinestyle = 'solid'
    
    CPMGColor = '#FF7F00'# Orange
    CPMGLinestyle = 'solid'

    noiseColor = '#2CA02C'# Green
    noiseLinestyle = 'dashdot'
    noiseLabel = r'$S(\nu)$'

    lwps = 3 # Linewidth for pulse sequence plots
    # Find max index for plotting
    cutoffIdx = len(noise)-1 # Initialize
    for i in range(len(noise)-1, -1, -1):
        if (noise[i] / freq[i] > 1e-3):
            cutoffIdx = i
            break
    print("cutoffFreq = {}".format(freq[cutoffIdx]))


    if title is not None:
        fig.suptitle(title)
    
    # Plot pulse sequence over time

    axd['A'].set_title('UDD ($N_{{pulse}} = {}$)'.format(int(nPulse)))
    axd['A'].sharex(axd['C'])
    axd['A'].vlines(UDDTime, 0, 1, color=UDDColor, linestyle=UDDLinestyle, linewidth = lwps)
    axd['A'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    #axd['A'].tick_params(labelbottom=False) # Remove x axis labels
    axd['A'].grid(axis = 'x')

    axd['B'].set_title('PDD ($N_{{pulse}} = {}$)'.format(int(nPulse)))
    axd['B'].sharex(axd['C'])
    axd['B'].vlines(PDDTime, 0, 1, color=PDDColor, linewidth = lwps)
    axd['B'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    #axd['B'].tick_params(labelbottom=False) # Remove x axis labels
    axd['B'].grid(axis = 'x')
   
    axd['C'].set_title('CPMG ($N_{{pulse}} = {}$)'.format(int(nPulse)))
    axd['C'].set_xlabel('Time')
    axd['C'].set_xlim(0, tMax)
    axd['C'].vlines(CPMGTime, 0, 1, color=CPMGColor, linestyle=CPMGLinestyle, linewidth=lwps)
    axd['C'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['C'].grid(axis = 'x')

    # Plot filter functions and noise PSD
    angfreq = 2 * np.pi * freq

    axd['D'].set_ylabel(r'$F(\omega)$')
    axd['D'].sharey(axd['F'])
    axUDDNoise = axd['D'].twinx()
    axUDDNoise.set_ylabel(noiseLabel)
    filterPlotUDD = axd['D'].plot(angfreq[:cutoffIdx], UDDFilter[:cutoffIdx], color=UDDColor, linestyle=UDDLinestyle, label = r'$F(\omega)_{UDD}$')
    noisePlotUDD = axUDDNoise.plot(angfreq[:cutoffIdx], noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\omega)$')
    linesUDD = filterPlotUDD + noisePlotUDD
    labelsUDD = [l.get_label() for l in linesUDD]
    axd['D'].tick_params(labelbottom=False) # Remove x axis labels
    axd['D'].set_title(UDDLabel)
    axd['D'].legend(linesUDD, labelsUDD)


    axd['E'].set_ylabel(r'$F(\omega)$')
    axd['E'].sharey(axd['F'])
    axPDDNoise = axd['E'].twinx()
    axPDDNoise.set_ylabel(noiseLabel)
    filterPlotPDD = axd['E'].plot(angfreq[:cutoffIdx], PDDFilter[:cutoffIdx], color=PDDColor, linestyle=PDDLinestyle, label = r'$F(\omega)_{PDD}$')
    noisePlotPDD = axPDDNoise.plot(angfreq[:cutoffIdx], noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\omega)$')
    linesPDD = filterPlotPDD + noisePlotPDD
    labelsPDD = [l.get_label() for l in linesPDD]
    axd['E'].tick_params(labelbottom=False) # Remove x axis labels
    axd['E'].set_title(PDDLabel)
    axd['E'].legend(linesPDD, labelsPDD)

    axd['F'].set_ylabel(r'$F(\omega)$')
    axCPMGNoise = axd['F'].twinx()
    axCPMGNoise.set_ylabel(noiseLabel)
    filterPlotCPMG = axd['F'].plot(angfreq[:cutoffIdx], CPMGFilter[:cutoffIdx], color=CPMGColor, linestyle=CPMGLinestyle, label = r'$F(\omega)_{CPMG}$')
    noisePlotCPMG = axCPMGNoise.plot(angfreq[:cutoffIdx], noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\omega)$')
    linesCPMG = filterPlotCPMG + noisePlotCPMG
    labelsCPMG = [l.get_label() for l in linesCPMG]
    axd['F'].set_title(CPMGLabel)
    axd['F'].set_xlabel(r'$\omega$')
    axd['F'].legend(linesCPMG, labelsCPMG)
    #axd['F'].set_yscale('log')

    axUDDNoise.sharey(axCPMGNoise)
    axPDDNoise.sharey(axCPMGNoise)
    #axCPMGNoise.set_yscale('log')

    plt.show()
