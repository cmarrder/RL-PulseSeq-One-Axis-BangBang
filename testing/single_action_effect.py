import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import pulse_sequence as ps
import harmonic_actions as ha

ymax = 0.2

def setup_xaxis(ax):
    """Set up common parameters for the Axes in the example."""
    # only show the bottom spine
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.spines['bottom'].set_position('center')

    # define tick positions
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.00))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    ax.xaxis.set_ticks_position('bottom')
    #ax.tick_params(which='major', width=1.00, length=5)
    #ax.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-ymax, ymax)

def plot_pulse_seq(pulse_times, linestyle='solid', xlabel=None):
    fig, ax = plt.subplots(figsize=(4, 2.5), layout='constrained')
    setup_xaxis(ax)
    tick_labels = ['0', 'T']#['0', '0.25T', '0.5T', '0.75T', 'T']
    tick_locs = [0, 1]#[0, 0.25, 0.5, 0.75, 1]

    ymax_frac = 0.4
    # Define pad sizes
    PAD_DEFAULT_AXIS_LABEL = 9
    # Define font sizes
    SIZE_TICK_LABEL = 32 
    SIZE_AXIS_LABEL = 32
    # Define linewidths
    LW_SEQUENCE = 6
    ax.set_xticks(tick_locs, labels=tick_labels)
    ax.tick_params(axis='x', which='major', labelsize=SIZE_TICK_LABEL) 
    ax.vlines(pulse_times, - ymax_frac * ymax, ymax_frac * ymax, linestyle=linestyle, linewidth=LW_SEQUENCE)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=SIZE_AXIS_LABEL, labelpad=PAD_DEFAULT_AXIS_LABEL)
    return fig

def plot_2_pulse_seq(pulse_times_1, pulse_times_2, xlabel=None):
    fig, ax = plt.subplots(figsize=(4, 2.5), layout='constrained')
    setup_xaxis(ax)
    tick_labels = ['0', 'T']#['0', '0.25T', '0.5T', '0.75T', 'T']
    tick_locs = [0, 1]#[0, 0.25, 0.5, 0.75, 1]

    ymax_frac = 0.4
    # Define pad sizes
    PAD_DEFAULT_AXIS_LABEL = 9
    # Define font sizes
    SIZE_TICK_LABEL = 32 
    SIZE_AXIS_LABEL = 32
    # Define linewidths
    LW_SEQUENCE = 6

    vlines_color = 'tab:blue'

    ax.set_xticks(tick_locs, labels=tick_labels)
    ax.tick_params(axis='x', which='major', labelsize=SIZE_TICK_LABEL) 
    ax.vlines(pulse_times_1, - ymax_frac * ymax, ymax_frac * ymax, linestyle=(0, (1, 0.4)), linewidth=LW_SEQUENCE, color=vlines_color)
    ax.vlines(pulse_times_2, - ymax_frac * ymax, ymax_frac * ymax, linestyle='solid', linewidth=LW_SEQUENCE, color=vlines_color)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=SIZE_AXIS_LABEL, labelpad=PAD_DEFAULT_AXIS_LABEL)
    return fig

def plot_old_new_filters(oldTimes, newTimes, maxTime, show=True, save=None):

    # Define font sizes
    SIZE_TICK_LABEL = 2*32
    SIZE_AXIS_LABEL = 2.5*32
    # Define linewidths
    LW_CURVE = 2*6

    curve_color = 'tab:blue'

    cpmgPeakFreq = ps.CPMGPeakFreq(len(oldTimes), maxTime)
    freqs = 2 * np.pi * np.linspace(0, 0.7*cpmgPeakFreq, 1000)

    oldFilter = ps.FilterFunc(freqs, oldTimes, maxTime)
    newFilter = ps.FilterFunc(freqs, newTimes, maxTime)

    fig, ax = plt.subplots(figsize=(18,8), layout='constrained')
    #fig, ax = plt.subplots(figsize=(18, 24), layout='constrained')
    ax.plot(freqs, oldFilter, color=curve_color, linestyle='dashed', label='Old Filter', lw=LW_CURVE)
    ax.plot(freqs, newFilter, color=curve_color, linestyle='solid', label='New Filter', lw=LW_CURVE)


    ax.tick_params(axis='both', which='major', labelsize=SIZE_TICK_LABEL) 
    ax.set_xlabel('$\omega$', fontsize=SIZE_AXIS_LABEL) 
    ax.set_ylabel('$F(\omega)$', fontsize=SIZE_AXIS_LABEL) 
    ax.legend(prop={'size': SIZE_AXIS_LABEL})

    if save is not None:
        plt.savefig(save, dpi=300)
    if show:
        plt.show()
    
    return

if __name__=='__main__':

    # matplotlib metaparameters 
    #mpl.rcParams['axes.linewidth'] = 12 # For filter plot
    mpl.rcParams['axes.linewidth'] = 6 # For pulse seq plots
    mpl.rcParams['xtick.major.size'] = 10#5
    mpl.rcParams['xtick.major.width'] = 6
    mpl.rcParams['xtick.minor.size'] = 1
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.major.size'] = 10#5
    mpl.rcParams['ytick.major.width'] = 6
    mpl.rcParams['ytick.minor.size'] = 1
    mpl.rcParams['ytick.minor.width'] = 1
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Droid Sans']

    nPulse = 4
    maxTime = 1
    j = -2
    eta = 0.9*ha.eta_exclusive_bounds(j, maxTime)[1]
    old_times = ps.PDD(nPulse, maxTime)
    new_times = ha.kappa(old_times, j, eta, maxTime)

    # MAKE INDIVIDUAL 1D PLOTS OF PULSE SEQ TIMES
    #fig = plot_pulse_seq(old_times, linestyle=(0, (1, 0.4)), xlabel='Old Times')
    #fig = plot_pulse_seq(new_times, linestyle='solid', xlabel='New Times')
    #plt.savefig('paper_plots/TimesOld.svg', dpi=300)
    #plt.savefig('paper_plots/TimesNew.svg', dpi=300)

    # MAKE SINGLE 1D PLOT WITH BOTH OLD AND NEW TIMES ON IT
    fig = plot_2_pulse_seq(old_times, new_times, xlabel='Move the Times')
    plt.savefig('paper_plots/AgentMovingTimes.svg', dpi=300)
    
    # MAKE FILTER FUNCTION PLOTS
    #plot_old_new_filters(old_times, new_times, maxTime, save='paper_plots/FiltersOldNew.svg')
