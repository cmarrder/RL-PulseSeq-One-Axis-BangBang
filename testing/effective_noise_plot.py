import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pulse_sequence as ps
import os

plotCurves = True

# Load data

oDir = '/home/charlie/Documents/ml/CollectiveAction/data'

nPulse = 8#int( np.loadtxt(os.path.join(oDir, 'nPulse.txt')) ) # Number of pulse applications
#nTimeStep = int( np.loadtxt(os.path.join(oDir, 'nTimeStep.txt')) )# Number of pulse chances/locations
tMax = np.loadtxt(os.path.join(oDir, 'maxTime.txt'))
param1 = np.loadtxt(os.path.join(oDir, 'noiseParam1.txt'))
param2 = np.loadtxt(os.path.join(oDir, 'noiseParam2.txt'))

freq = np.loadtxt(os.path.join(oDir, 'freq.txt'))
#recip = np.reciprocal(freq, where=freq >= 1e-8) #ps.fermi_dirac(freq, 20/2/np.pi, 0.1) #1 / (1 + freq**2)#np.loadtxt(os.path.join(oDir, 'sOmega.txt'))
#sOmega = np.loadtxt(os.path.join(oDir, 'sOmega.txt'))

#sOmega = np.where(recip < 1e-8, 1e8, recip)

# Choose noise PSD profile

cpmgPeak = nPulse / 2 / tMax
param1 = cpmgPeak / 8
param2 = 0.4#0.01
# Fermi-Dirac distribution for studying hard to smooth cutoff (low temp to high temp)
#S = ps.fermi_dirac(freq, param1, param2)
height = 1
fwhm = 0.5
noise = height * (fwhm/2)**2 / (freq**2 + (fwhm/2)**2)

# Find max index for plotting
cutoffIdx = len(noise)-1 # Initialize
for i in range(len(noise)-1, -1, -1):
    #if (noise[i] / freq[i] > 1e-5):
    if (noise[i] / freq[i] > 1e-5):
        cutoffIdx = i
        break

negFreq = np.flip(-freq[1:cutoffIdx])
freq = np.concatenate((negFreq, freq[1:cutoffIdx]))
print(freq)
noise = height * (fwhm/2)**2 / (freq**2 + (fwhm/2)**2)

# Sum of Lorentzians peaked at the CPMG filter function peaks so we make it hard for CPMG to perform well
#tau = tMax / (2 * nPulse)
#centers = (np.arange(nPulse) + 0.5) / (2 * tau)
#fwhms = np.ones_like(centers) / 2
#S = ps.lorentzians(freq, centers, fwhms)
## Normalize if you wish:
#S /= len(centers)

# Crunch numbers
CPMGTime = ps.CPMG(nPulse, tMax)
print("CPMG")
print(CPMGTime)

if plotCurves == True:


    # Define font sizes
    SIZE_TICK_LABEL = 16
    #SIZE_AXIS_LABEL = 16
    #SIZE_TITLE = 22
    # Define pad sizes
    PAD_AXH = 18
    PAD_DEFAULT_AXIS_LABEL = 14
    PAD_TITLE = 18

    # matplotlib metaparameters 
    mpl.rcParams['axes.linewidth'] = 2 * 1.5
    mpl.rcParams['xtick.major.size'] = 7#5
    mpl.rcParams['xtick.major.width'] = 2 
    mpl.rcParams['xtick.minor.size'] = 1
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.major.size'] = 7#5
    mpl.rcParams['ytick.major.width'] = 2 
    mpl.rcParams['ytick.minor.size'] = 1
    mpl.rcParams['ytick.minor.width'] = 1
    mpl.rcParams['font.size'] = 16
#    mpl.rcParams['axes.titlepad'] = PAD_AXH

    CPMGFilter = ps.FilterFunc(freq, CPMGTime, tMax)
    
    CPMGOverlap = ps.chi(freq, noise, CPMGFilter)

    CPMGFid = ps.fidelity(CPMGOverlap)

    print('\nFIDELITIES')
    print('CPMG Fidelity')
    print(CPMGFid)

    sequence_title = 'CPMG ($N = {}$)'.format(int(nPulse))
    CPMGLabel = sequence_title
    
    string_Na = '$N_{{pulse}} = {}$, '.format(nPulse)
    string_mu = '$\mu$ = {:.3e}, '.format(param1)
    string_temp = '$T$ = {:.3e}'.format(param2)
    #title = string_Na + string_mu + string_temp
    title = 'Performance of UDD, PDD, and CPMG on $1/f$ Noise'
    
    
    fig = plt.figure(layout = 'constrained', figsize = (9, 6))
    mosaic = """
             D
             E
             """
    axd = fig.subplot_mosaic(mosaic)
    
    CPMGColor = '#FF7F00'# Orange
    CPMGLinestyle = 'solid'

    noiseColor = '#2CA02C'# Green
    noiseLinestyle = 'dashdot'
    noiseLabel = r'$S(\omega)$'

    # Set linewidths for different plot lines
    LW_SEQUENCE = 2.5 * 1.5
    LW_CURVES = 1.75 * 1.5

    angfreq = 2 * np.pi * freq
    ff = CPMGFilter
    sequence = CPMGTime

    time_tick_labels = ['0', '0.2T', '0.4T', '0.6T', '0.8T', 'T']
    time_tick_locs = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ins_axlist = []

    inset_x = 0.05#0.3
    inset_y = 0.65#0.80
    inset_w = 0.4
    inset_h = 0.15
    ax_seq = axd['D'].inset_axes([inset_x, inset_y, inset_w, inset_h])
    axd['E'].sharex(axd['D'])
    # Plot pulse sequence times
    ax_seq.set_title(sequence_title)
    ax_seq.vlines(sequence, 0, 1, color=CPMGColor, linestyle=CPMGLinestyle, linewidth = LW_SEQUENCE)
    ax_seq.yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    ax_seq.set_xticks(ticks=time_tick_locs, labels=time_tick_labels)
    ax_seq.tick_params(axis='x', which='major', labelsize=0.8*SIZE_TICK_LABEL) 
    ax_seq.set_xlabel('Time')
    # Plot filters and noise PSD
    axd['D'].plot(angfreq, ff, color=CPMGColor, linestyle=CPMGLinestyle, label = r'$F(\omega)$', linewidth=LW_CURVES)
    axd['D'].plot(angfreq, noise, color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\omega)$', linewidth=LW_CURVES)
    axd['D'].tick_params(axis='both', which='major', labelsize=SIZE_TICK_LABEL) 
    axd['D'].legend(prop={'size': SIZE_TICK_LABEL})
    axd['E'].plot(angfreq, ff*noise, color=CPMGColor, linestyle=CPMGLinestyle, label = r'$S(\omega) \cdot F(\omega)$', linewidth=LW_CURVES)
    axd['E'].tick_params(axis='both', which='major', labelsize=SIZE_TICK_LABEL) 
    # Format y-axis to scientific notation
    axd['E'].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axd['E'].legend(prop={'size': SIZE_TICK_LABEL})
    axd['E'].set_title('Effective Noise Spectrum')
    #axd['E'].set_yscale('log')

    axd['E'].set_xlabel(r'$\omega$')
    #axd['D'].set_ylim(bottom=1e-12, top=1e9)
    ax_seq.set_xlim(0, tMax)
    ax_seq.set_ylim(0, 1)

    plt.savefig('../paper_plots/SampleEffectiveNoise.png', dpi=300)
    plt.show()
