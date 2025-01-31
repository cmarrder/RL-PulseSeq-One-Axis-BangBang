import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm, colors
import pulse_sequence as ps
import os

def kappa(t, harmonic, eta, max_time):
    return t + eta * np.sin(np.pi * harmonic * t / max_time)

#def action_sequence_plot(initial_times,
#                         action_sequence,
#                         harmonic_set,
#                         eta_set,
#                         max_time):
#    """
#    PARAMETERS:
#    initial_times (list like)
#    action_sequence (list)
#    harmonic_set (list)
#    eta_set (list)
#    max_time (float)
#    """
#
#    # Define font sizes
#    SIZE_DEFAULT = 16
#    SIZE_LARGE = 22
#    # Define pad sizes
#    PAD_DEFAULT = 10
#    # matplotlib metaparameters 
#    mpl.rcParams['axes.linewidth'] = 2
#    mpl.rcParams['xtick.major.size'] = 7#5
#    mpl.rcParams['xtick.major.width'] = 2 
#    mpl.rcParams['xtick.minor.size'] = 1
#    mpl.rcParams['xtick.minor.width'] = 1
#    mpl.rcParams['ytick.major.size'] = 7#5
#    mpl.rcParams['ytick.major.width'] = 2 
#    mpl.rcParams['ytick.minor.size'] = 1
#    mpl.rcParams['ytick.minor.width'] = 1
#    mpl.rcParams['axes.titlepad'] = PAD_DEFAULT
#    plt.rcParams['figure.constrained_layout.use'] = True
#    pulse_seq_lw = 2 # Linewidth for pulse sequences in the plot.
#
#    fig, ax = plt.subplots()
#
#    plt.xlim( (0, max_time) )
#    xtick_positions = [0, max_time/4, max_time/2, 3*max_time/4, max_time]
#    xtick_labels = ['0', '0.25T', '0.5T', '0.75T', 'T']
#    plt.xticks(ticks=xtick_positions, labels=xtick_labels)
#
#    Nstep = len(action_sequence)
#    Nharmonics = len(harmonic_set)
#
#    vlines_ymin = np.arange(Nstep + 1)
#    vlines_ymax = vlines_ymin + 1
#
#    ytick_locs = vlines_ymin + 0.5
#    ytick_labels = np.arange(Nstep + 1)
#    plt.yticks(ticks=ytick_locs, labels = ytick_labels)
#
#    # Make colorbar. Get color map, normalize colors, set ticks and labels to be in between
#    # different colors.
#    cmap = plt.get_cmap('tab10', Nharmonics + 1)
#    norm = colors.Normalize(0, Nharmonics + 1)
#    cbar_tick_locs = 0.5 + np.arange(Nharmonics + 1)
#    cbar_tick_labels = ['None'] + [str(j) for j in harmonic_set]
#    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
#    cbar.set_ticks(cbar_tick_locs)
#    cbar.set_ticklabels(cbar_tick_labels)
#
#    # Plot pulse sequence vertical lines for the initial state and then each subsequent state.
#    step = 0
#    pulse_times = np.copy(initial_times)
#    plt.vlines(pulse_times, vlines_ymin[step], vlines_ymax[step], color = cmap(norm(0)), lw=pulse_seq_lw)
#    for action in action_sequence:
#        step += 1
#        pulse_times = kappa(pulse_times, harmonic_set[action], eta_set[action], max_time)
#        plt.vlines(pulse_times, vlines_ymin[step], vlines_ymax[step], color = cmap(norm(action + 1)), lw=pulse_seq_lw)
#    plt.show()
#    return


def action_sequence_plot(initial_times,
                         action_sequence,
                         harmonic_set,
                         eta_set,
                         max_time):
    """
    PARAMETERS:
    initial_times (list like)
    action_sequence (list)
    harmonic_set (list)
    eta_set (list)
    max_time (float)
    """

    # Define font sizes
    SIZE_DEFAULT = 16
    SIZE_LARGE = 22
    # Define pad sizes
    PAD_DEFAULT = 10
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
    mpl.rcParams['axes.titlepad'] = PAD_DEFAULT
    plt.rcParams['figure.constrained_layout.use'] = True
    pulse_seq_lw = 2 # Linewidth for pulse sequences in the plot.
    agent_color = '#984EA3'# Lilac

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.grid(axis='y')

    # X axis
    plt.xlim( (0, max_time) )
    xtick_positions = [0, max_time/4, max_time/2, 3*max_time/4, max_time]
    xtick_labels = ['0', '0.25T', '0.5T', '0.75T', 'T']
    plt.xticks(ticks=xtick_positions, labels=xtick_labels)
    ax1.set_xlabel('Time')

    # Parameters for vlines
    Nstep = len(action_sequence)
    vlines_ymin = np.arange(Nstep + 1)
    vlines_ymax = vlines_ymin + 1

    # Lefthand Y axis
    ax1.set_ylim((-1, Nstep + 2))
    y1tick_locs = vlines_ymin + 0.5
    y1tick_labels = np.arange(Nstep + 1)
    ax1.set_yticks(ticks=y1tick_locs, labels = y1tick_labels)
    ax1.set_ylabel('Steps')

    # Righthand Y axis
    ax2.set_ylim((-1, Nstep + 2))
    y2tick_locs = y1tick_locs[1:] 
    y2tick_labels = [harmonic_set[a] for a in action_sequence]
    ax2.set_yticks(ticks=y2tick_locs, labels = y2tick_labels)
    ax2.set_ylabel('Harmonics Chosen')


    # Plot pulse sequence vertical lines for the initial state and then each subsequent state.
    step = 0
    pulse_times = np.copy(initial_times)
    ax1.vlines(pulse_times, vlines_ymin[step], vlines_ymax[step], color = agent_color, lw=pulse_seq_lw)
    for action in action_sequence:
        step += 1
        pulse_times = kappa(pulse_times, harmonic_set[action], eta_set[action], max_time)
        ax1.vlines(pulse_times, vlines_ymin[step], vlines_ymax[step], color = agent_color, lw=pulse_seq_lw)
    plt.title('Effect of Actions on Pulse Sequence Timings')
    plt.show()
    return

if __name__=="__main__":
    eta1 = 0.04
    eta2 = 0.02
    eta4 = 0.01
    eta8 = 0.005
    harmonic_set = [1, -1, 4, -4, 8, -8]
    eta_set = [eta1, eta1, eta4, eta4, eta8, eta8]

    job_dir = '/home/charlie/Documents/ml/CollectiveAction/data/job_00000'
    action_data = os.path.join(job_dir, 'action.txt')
    save_dir = '/home/charlie/Documents/ml/CollectiveAction/action_seq'
    action_sequence = list(np.loadtxt(action_data, dtype=int))
    
    max_time = 1
    Npulse = 8
    pulse_times = ps.PDD(Npulse, max_time)

    action_sequence_plot(pulse_times, action_sequence, harmonic_set, eta_set, max_time)

    #agent_color = '#984EA3'# Lilac

    #plt.vlines(pulse_times, 0, 1, color=agent_color)
    #plt.title('Initial State')
    #plt.savefig(os.path.join(save_dir, 'step_{:03}.png'.format(0)))

    #for step, action in enumerate(action_sequence):
    #   pulse_times = kappa(pulse_times, harmonic_set[action], eta_set[action], max_time)
    #   plt.vlines(pulse_times, 0, 1, color=agent_color)
    #   plt.title('Apply $\kappa_{{{}}}$'.format(harmonic_set[action]))
    #   plt.savefig(os.path.join(save_dir, 'step_{:03}.png'.format(step+1)))
