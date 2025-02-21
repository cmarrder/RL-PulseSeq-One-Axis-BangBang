import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pulse_sequence as ps
import os

plotCurves = True

# Load data

oDir = '/home/charlie/Documents/ml/CollectiveAction/eta_scan_data/1_over_f/harmonics_01_02_08/job_00002/run_00001'

nPulse = 8#int( np.loadtxt(os.path.join(oDir, 'nPulse.txt')) ) # Number of pulse applications
#nTimeStep = int( np.loadtxt(os.path.join(oDir, 'nTimeStep.txt')) )# Number of pulse chances/locations
tMax = np.loadtxt(os.path.join(oDir, 'maxTime.txt'))
param1 = np.loadtxt(os.path.join(oDir, 'noiseParam1.txt'))
param2 = np.loadtxt(os.path.join(oDir, 'noiseParam2.txt'))

freq = np.loadtxt(os.path.join(oDir, 'freq.txt'))
recip = np.reciprocal(freq, where=freq >= 1e-8)#np.loadtxt(os.path.join(oDir, 'sOmega.txt')) #ps.fermi_dirac(freq, 20/2/np.pi, 0.1) #1 / (1 + freq**2)#np.loadtxt(os.path.join(oDir, 'sOmega.txt'))
sOmega = np.where(recip < 1e-8, 1e8, recip)

#finalState = np.loadtxt(os.path.join(oDir, 'state.txt'))
#reward = np.loadtxt(os.path.join(oDir, 'reward.txt'))
#loss = np.loadtxt(os.path.join(oDir, 'loss.txt'))

# Choose noise PSD profile

cpmgPeak = nPulse / 2 / tMax
param1 = cpmgPeak / 8
param2 = 0.4#0.01
# Fermi-Dirac distribution for studying hard to smooth cutoff (low temp to high temp)
#S = ps.fermi_dirac(freq, param1, param2)

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
print("\nPULSE TIMES")
print("\nUDD")
print(UDDTime)
print("PDD")
print(PDDTime)
print("CPMG")
print(CPMGTime)

#UDDSteps = np.diff(UDDTime)
#CPMGSteps = np.diff(CPMGTime)


if plotCurves == True:


    # Define font sizes
    SIZE_TICK_LABEL = 12
    SIZE_AXIS_LABEL = 16
    SIZE_TITLE = 22
    # Define pad sizes
    PAD_AXH = 18
    PAD_DEFAULT_AXIS_LABEL = 14
    PAD_TITLE = 18

    # matplotlib metaparameters 
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['xtick.major.size'] = 7#5
    mpl.rcParams['xtick.major.width'] = 2 
    mpl.rcParams['xtick.minor.size'] = 1
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.major.size'] = 7#5
    mpl.rcParams['ytick.major.width'] = 2 
    mpl.rcParams['ytick.minor.size'] = 1
    mpl.rcParams['ytick.minor.width'] = 1
#    mpl.rcParams['axes.titlepad'] = PAD_AXH

    UDDFilter = ps.FilterFunc(freq, UDDTime, tMax)
    PDDFilter = ps.FilterFunc(freq, PDDTime, tMax)
    CPMGFilter = ps.FilterFunc(freq, CPMGTime, tMax)
    
    UDDOverlap = ps.chi(freq, noise, UDDFilter)
    PDDOverlap = ps.chi(freq, noise, PDDFilter)
    CPMGOverlap = ps.chi(freq, noise, CPMGFilter)

    UDDFid = ps.fidelity(UDDOverlap)
    PDDFid = ps.fidelity(PDDOverlap)
    CPMGFid = ps.fidelity(CPMGOverlap)

    print('\nFIDELITIES')
    print('\nUDD Fidelity')
    print(UDDFid)
    print('PDD Fidelity')
    print(PDDFid)
    print('CPMG Fidelity')
    print(CPMGFid)

    print('\nNORMALIZATION')
    dfreq = freq[1] - freq[0]
    print('\nIntegral of UDD Filter from -inf to inf')
    print(2 * np.sum(UDDFilter) * dfreq)
    print('Integral of PDD Filter from -inf to inf')
    print(2 * np.sum(PDDFilter) * dfreq)
    print('Integral of CPMG Filter from -inf to inf')
    print(2 * np.sum(CPMGFilter) * dfreq)

    sequence_titles = ['UDD ($N = {}$)'.format(int(nPulse)), 'PDD ($N = {}$)'.format(int(nPulse)), 'CPMG ($N = {}$)'.format(int(nPulse))]
    UDDLabel = 'UDD ($N = {0}$), $p = {1:.3f}$'.format(nPulse, UDDFid)
    PDDLabel = 'PDD ($N = {0}$), $p = {1:.3f}$'.format(nPulse, PDDFid)
    CPMGLabel = 'CPMG ($N = {0}$), $p = {1:.3f}$'.format(nPulse, CPMGFid)
    
    string_Na = '$N_{{pulse}} = {}$, '.format(nPulse)
    string_mu = '$\mu$ = {:.3e}, '.format(param1)
    string_temp = '$T$ = {:.3e}'.format(param2)
    #title = string_Na + string_mu + string_temp
    title = 'Performance of UDD, PDD, and CPMG on $1/f$ Noise'
    
    # Find max index for plotting
    cutoffIdx = len(noise)-1 # Initialize
    for i in range(len(noise)-1, -1, -1):
        #if (noise[i] / freq[i] > 1e-5):
        if (noise[i] / freq[i] > 1e-3):
            cutoffIdx = i
            break
    
# PLOTS WITH TWIN AXES FOR FILTERS AND NOISE SPECTRA
#    fig = plt.figure(layout = 'constrained', figsize = (8, 8))
#    mosaic = """
#             AD
#             BE
#             CF
#             """
#    axd = fig.subplot_mosaic(mosaic, width_ratios=[1, 2])
#    
#    UDDColor = '#377EB8'# Blue
#    UDDLinestyle = 'solid'
#
#    PDDColor = '#984EA3'# Lilac
#    PDDLinestyle = 'solid'
#    
#    CPMGColor = '#FF7F00'# Orange
#    CPMGLinestyle = 'solid'
#
#    noiseColor = '#2CA02C'# Green
#    noiseLinestyle = 'dashdot'
#    noiseLabel = r'$S(\omega)$'
#
#    # Set linewidths for different plot lines
#    LW_SEQUENCE = 3
#    LW_CURVES = 3
#
#    # Find max index for plotting
#    cutoffIdx = len(noise)-1 # Initialize
#    for i in range(len(noise)-1, -1, -1):
#        if (noise[i] / freq[i] > 1e-3):
#            cutoffIdx = i
#            break
#    print("cutoffFreq = {}".format(freq[cutoffIdx]))
#
#
#    if title is not None:
#        fig.suptitle(title, fontsize=SIZE_TITLE)
    ############## PLOT EACH PULSE SEQUENCE OVER TIME ############## 
    ################################################################

#    time_tick_labels = ['0', '0.2T', '0.4T', '0.6T', '0.8T', 'T']
#    time_tick_locs = [0, 0.2, 0.4, 0.6, 0.8, 1]
#
#    #axd['A'].set_title('UDD ($N_{{pulse}} = {}$)'.format(int(nPulse)))
#    axd['A'].set_title('UDD ($N = {}$)'.format(int(nPulse)))
#    axd['A'].sharex(axd['C'])
#    axd['A'].vlines(UDDTime, 0, 1, color=UDDColor, linestyle=UDDLinestyle, linewidth = LW_SEQUENCE)
#    axd['A'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
#    #axd['A'].tick_params(labelbottom=False) # Remove x axis labels
#    axd['A'].set_xticks(ticks=time_tick_locs, labels=time_tick_labels)
#    axd['A'].grid(axis = 'x')
#
#    #axd['B'].set_title('PDD ($N_{{pulse}} = {}$)'.format(int(nPulse)))
#    axd['B'].set_title('PDD ($N = {}$)'.format(int(nPulse)))
#    axd['B'].sharex(axd['C'])
#    axd['B'].vlines(PDDTime, 0, 1, color=PDDColor, linewidth = LW_SEQUENCE)
#    axd['B'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
#    #axd['B'].tick_params(labelbottom=False) # Remove x axis labels
#    axd['B'].grid(axis = 'x')
#   
#    #axd['C'].set_title('CPMG ($N_{{pulse}} = {}$)'.format(int(nPulse)))
#    axd['C'].set_title('CPMG ($N = {}$)'.format(int(nPulse)))
#    axd['C'].set_xlabel('Time')
#    axd['C'].set_xlim(0, tMax)
#    axd['C'].vlines(CPMGTime, 0, 1, color=CPMGColor, linestyle=CPMGLinestyle, linewidth=LW_SEQUENCE)
#    axd['C'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
#    axd['C'].grid(axis = 'x')

    ################# PLOT FILTER FUNCTIONS AND NOISE PSD #################
    #######################################################################

#    angfreq = 2 * np.pi * freq

#    axd['D'].set_ylabel(r'$F(\omega)$')
#    axd['D'].sharey(axd['F'])
#    axUDDNoise = axd['D'].twinx()
#    axUDDNoise.set_ylabel(noiseLabel)
#    filterPlotUDD = axd['D'].plot(angfreq[:cutoffIdx], UDDFilter[:cutoffIdx], color=UDDColor, linestyle=UDDLinestyle, label = r'$F(\omega)_{UDD}$')
#    noisePlotUDD = axUDDNoise.plot(angfreq[:cutoffIdx], noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\omega)$')
#    linesUDD = filterPlotUDD + noisePlotUDD
#    labelsUDD = [l.get_label() for l in linesUDD]
#    #axd['D'].tick_params(labelbottom=False) # Remove x axis labels
#    axd['D'].set_yscale('log')
#    axUDDNoise.set_yscale('log')
#    axd['D'].set_title(UDDLabel)
#    axd['D'].legend(linesUDD, labelsUDD)
#
#
#    axd['E'].set_ylabel(r'$F(\omega)$')
#    axd['E'].sharey(axd['F'])
#    axPDDNoise = axd['E'].twinx()
#    axPDDNoise.set_ylabel(noiseLabel)
#    filterPlotPDD = axd['E'].plot(angfreq[:cutoffIdx], PDDFilter[:cutoffIdx], color=PDDColor, linestyle=PDDLinestyle, label = r'$F(\omega)_{PDD}$')
#    noisePlotPDD = axPDDNoise.plot(angfreq[:cutoffIdx], noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\omega)$')
#    linesPDD = filterPlotPDD + noisePlotPDD
#    labelsPDD = [l.get_label() for l in linesPDD]
#    #axd['E'].tick_params(labelbottom=False) # Remove x axis labels
#    axd['E'].set_yscale('log')
#    axPDDNoise.set_yscale('log')
#    axd['E'].set_title(PDDLabel)
#    axd['E'].legend(linesPDD, labelsPDD)
#
#    axd['F'].set_ylabel(r'$F(\omega)$')
#    axCPMGNoise = axd['F'].twinx()
#    axCPMGNoise.set_ylabel(noiseLabel)
#    filterPlotCPMG = axd['F'].plot(angfreq[:cutoffIdx], CPMGFilter[:cutoffIdx], color=CPMGColor, linestyle=CPMGLinestyle, label = r'$F(\omega)_{CPMG}$')
#    noisePlotCPMG = axCPMGNoise.plot(angfreq[:cutoffIdx], noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\omega)$')
#    linesCPMG = filterPlotCPMG + noisePlotCPMG
#    labelsCPMG = [l.get_label() for l in linesCPMG]
#    axd['F'].set_title(CPMGLabel)
#    axd['F'].set_xlabel(r'$\omega$')
#    axd['F'].set_yscale('log')
#    axCPMGNoise.set_yscale('log')
#    axd['F'].legend(linesCPMG, labelsCPMG)
#    #axd['F'].set_yscale('log')
#
#    axUDDNoise.sharey(axCPMGNoise)
#    axPDDNoise.sharey(axCPMGNoise)
#    #axCPMGNoise.set_yscale('log')


############## PLOTS WITH FILTER AND NOISE SPECTRA ON SAME LOG SCALE


#    fig = plt.figure(layout = 'constrained', figsize = (8, 8))
#    mosaic = """
#             AD
#             BE
#             CF
#             """
#    axd = fig.subplot_mosaic(mosaic, width_ratios=[1, 2])
#    
#    UDDColor = '#377EB8'# Blue
#    UDDLinestyle = 'solid'
#
#    PDDColor = '#984EA3'# Lilac
#    PDDLinestyle = 'solid'
#    
#    CPMGColor = '#FF7F00'# Orange
#    CPMGLinestyle = 'solid'
#
#    noiseColor = '#2CA02C'# Green
#    noiseLinestyle = 'dashdot'
#    noiseLabel = r'$S(\omega)$'
#
#    # Set linewidths for different plot lines
#    LW_SEQUENCE = 3
#    LW_CURVES = 3
#
#    # Find max index for plotting
#    cutoffIdx = len(noise)-1 # Initialize
#    for i in range(len(noise)-1, -1, -1):
#        if (noise[i] / freq[i] > 1e-3):
#            cutoffIdx = i
#            break
#    print("cutoffFreq = {}".format(freq[cutoffIdx]))
#
#
#    if title is not None:
#        fig.suptitle(title, fontsize=SIZE_TITLE)
#    angfreq = 2 * np.pi * freq[:cutoffIdx]
#    filters = np.stack((UDDFilter[:cutoffIdx], PDDFilter[:cutoffIdx], CPMGFilter[:cutoffIdx]))
#    sequences = np.stack((UDDTime, PDDTime, CPMGTime))
#
#    time_tick_labels = ['0', '0.2T', '0.4T', '0.6T', '0.8T', 'T']
#    time_tick_locs = [0, 0.2, 0.4, 0.6, 0.8, 1]
#    filter_linestyles = [UDDLinestyle, PDDLinestyle, CPMGLinestyle]
#    filter_colors = [UDDColor, PDDColor, CPMGColor]
#    filter_labels = [UDDLabel, PDDLabel, CPMGLabel]
#    filter_keys = ['D', 'E', 'F']
#    sequence_keys = ['A', 'B', 'C']
#    sequence_titles = ['UDD ($N = {}$)'.format(int(nPulse)), 'PDD ($N = {}$)'.format(int(nPulse)), 'CPMG ($N = {}$)'.format(int(nPulse))]
#
#    for i in range(3):
#        if i < 2:
#            axd[filter_keys[i]].sharey(axd['F'])
#            axd[filter_keys[i]].sharex(axd['F'])
#            axd[sequence_keys[i]].sharex(axd['C'])
#            axd[sequence_keys[i]].sharey(axd['C'])
#            #axd[sequence_keys[i]].tick_params(labelbottom=False) # Remove x axis labels
#        # Plot pulse sequence times
#        axd[sequence_keys[i]].set_title(sequence_titles[i], fontsize=SIZE_AXIS_LABEL)
#        axd[sequence_keys[i]].vlines(sequences[i], 0, 1, color=filter_colors[i], linestyle=filter_linestyles[i], linewidth = LW_SEQUENCE)
#        axd[sequence_keys[i]].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
#        axd[sequence_keys[i]].set_xticks(ticks=time_tick_locs, labels=time_tick_labels)
#        axd[sequence_keys[i]].tick_params(axis='x', which='major', labelsize=SIZE_TICK_LABEL) 
#        axd[sequence_keys[i]].grid(axis = 'x')
#        # Plot filters and noise PSD
#        axd[filter_keys[i]].plot(angfreq, filters[i], color=filter_colors[i], linestyle=filter_linestyles[i], label = r'$F(\omega)$')
#        axd[filter_keys[i]].plot(angfreq, noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\omega)$')
#        axd[filter_keys[i]].tick_params(axis='both', which='major', labelsize=SIZE_TICK_LABEL) 
#        #axd[filter_keys[i]].tick_params(labelbottom=False) # Remove x axis labels
#        axd[filter_keys[i]].set_yscale('log')
#        axd[filter_keys[i]].set_title(filter_labels[i], fontsize=SIZE_AXIS_LABEL)
#        axd[filter_keys[i]].legend(prop={'size': SIZE_TICK_LABEL})
#
#    axd['F'].set_xlabel(r'$\omega$', fontsize=SIZE_AXIS_LABEL)
#    axd['C'].set_xlabel('Time', fontsize=SIZE_AXIS_LABEL)
#    axd['C'].set_xlim(0, tMax)
#
##    plt.savefig('../paper_plots/Figure1.svg', dpi=300)
#    plt.show()



    fig = plt.figure(layout = 'constrained', figsize = (6, 8))
    mosaic = """
             D
             E
             F
             """
    axd = fig.subplot_mosaic(mosaic)
    
    UDDColor = '#377EB8'# Blue
    UDDLinestyle = 'solid'

    PDDColor = '#D62728'# Red
    PDDLinestyle = 'solid'
    
    CPMGColor = '#FF7F00'# Orange
    CPMGLinestyle = 'solid'

    noiseColor = '#2CA02C'# Green
    noiseLinestyle = 'dashdot'
    noiseLabel = r'$S(\omega)$'

    # Set linewidths for different plot lines
    LW_SEQUENCE = 2.5
    LW_CURVES = 1.75

    angfreq = 2 * np.pi * freq[:cutoffIdx]
    filters = np.stack((UDDFilter[:cutoffIdx], PDDFilter[:cutoffIdx], CPMGFilter[:cutoffIdx]))
    sequences = np.stack((UDDTime, PDDTime, CPMGTime))

    time_tick_labels = ['0', '0.2T', '0.4T', '0.6T', '0.8T', 'T']
    time_tick_locs = [0, 0.2, 0.4, 0.6, 0.8, 1]
    filter_linestyles = [UDDLinestyle, PDDLinestyle, CPMGLinestyle]
    filter_colors = [UDDColor, PDDColor, CPMGColor]
    filter_labels = [UDDLabel, PDDLabel, CPMGLabel]
    filter_keys = ['D', 'E', 'F']
    ins_axlist = []

    for i in range(3):
        inset_x = 0.3#0.55
        inset_y = 0.80#0.55
        inset_w = 0.4
        inset_h = 0.15
        ins_axlist.append(axd[filter_keys[i]].inset_axes([inset_x, inset_y, inset_w, inset_h]))
        if i > 0:
            axd[filter_keys[i]].sharey(axd['D'])
            axd[filter_keys[i]].sharex(axd['D'])
            ins_axlist[i].sharex(ins_axlist[0])
            ins_axlist[i].sharey(ins_axlist[0])
        # Plot pulse sequence times
        #ins_axlist[i].set_title(sequence_titles[i], fontsize=0.8*SIZE_AXIS_LABEL)
        ins_axlist[i].vlines(sequences[i], 0, 1, color=filter_colors[i], linestyle=filter_linestyles[i], linewidth = LW_SEQUENCE)
        ins_axlist[i].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
        ins_axlist[i].set_xticks(ticks=time_tick_locs, labels=time_tick_labels)
        ins_axlist[i].tick_params(axis='x', which='major', labelsize=0.8*SIZE_TICK_LABEL) 
        ins_axlist[i].set_xlabel('Time', fontsize=0.7*SIZE_AXIS_LABEL)
        #ins_axlist[i].grid(axis = 'x')
        # Plot filters and noise PSD
        axd[filter_keys[i]].plot(angfreq, filters[i], color=filter_colors[i], linestyle=filter_linestyles[i], label = r'$F(\omega)$', linewidth=LW_CURVES)
        axd[filter_keys[i]].plot(angfreq, noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\omega)$', linewidth=LW_CURVES)
        axd[filter_keys[i]].tick_params(axis='both', which='major', labelsize=SIZE_TICK_LABEL) 
        axd[filter_keys[i]].set_yscale('log')
        axd[filter_keys[i]].set_title(filter_labels[i], fontsize=SIZE_AXIS_LABEL)
        axd[filter_keys[i]].legend(prop={'size': SIZE_TICK_LABEL})

    axd['F'].set_xlabel(r'$\omega$', fontsize=SIZE_AXIS_LABEL)
    axd['D'].set_ylim(bottom=1e-12, top=1e9)
    ins_axlist[2].set_xlim(0, tMax)
    ins_axlist[2].set_ylim(0, 1)

#    plt.savefig('../paper_plots/Figure1.svg', dpi=300)
    plt.show()
