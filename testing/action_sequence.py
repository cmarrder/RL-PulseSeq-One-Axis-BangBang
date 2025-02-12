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

#def action_sequence_plot(initial_times,
#                         action_sequence,
#                         harmonic_set,
#                         eta_set,
#                         max_time,
#                         show=True,
#                         save=None):
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
#    agent_color = '#984EA3'# Lilac
#
#    fig = plt.figure(layout = 'constrained', figsize=(6, 12))
#    mosaic = """
#             P
#             """
#    axd = fig.subplot_mosaic(mosaic)
#    #fig, ax1 = plt.subplots()
#    #ax2 = ax1.twinx()
#    #ax1.grid(axis='y')
#    axH = axd['P'].twinx()
#    axd['P'].grid(axis='y')
#
#    # X axis
#    axd['P'].set_xlim( (0, max_time) )
#    xtick_positions = [0, max_time/4, max_time/2, 3*max_time/4, max_time]
#    xtick_labels = ['0', '0.25T', '0.5T', '0.75T', 'T']
#    axd['P'].set_xticks(ticks=xtick_positions, labels=xtick_labels)
#    axd['P'].set_xlabel('Time')
#
#    # Parameters for vlines
#    Nstep = len(action_sequence)
#    vlines_ymin = np.arange(Nstep + 1)
#    vlines_ymax = vlines_ymin + 1
#
#    # Lefthand Y axis
#    axd['P'].set_ylim((-1, Nstep + 2))
#    y1tick_locs = vlines_ymin + 0.5
#    y1tick_labels = np.arange(Nstep + 1)
#    axd['P'].set_yticks(ticks=y1tick_locs, labels = y1tick_labels)
#    axd['P'].set_ylabel('Steps')
#
#    # Righthand Y axis
#    axH.set_ylim((-1, Nstep + 2))
#    y2tick_locs = y1tick_locs[1:] 
#    y2tick_labels = [harmonic_set[a] for a in action_sequence]
#    axH.set_yticks(ticks=y2tick_locs, labels = y2tick_labels)
#    axH.set_ylabel('Harmonics Chosen')
#
#
#    # Plot pulse sequence vertical lines for the initial state and then each subsequent state.
#    step = 0
#    pulse_times = np.copy(initial_times)
#    axd['P'].vlines(pulse_times, vlines_ymin[step], vlines_ymax[step], color = agent_color, lw=pulse_seq_lw) # Plot initial state
#    for action in action_sequence:
#        step += 1
#        pulse_times = kappa(pulse_times, harmonic_set[action], eta_set[action], max_time)
#        axd['P'].vlines(pulse_times, vlines_ymin[step], vlines_ymax[step], color = agent_color, lw=pulse_seq_lw)
#    plt.title('Effect of Actions on Pulse Sequence Timings')
#    if show is True:
#        plt.show()
#    if save is not None:
#        plt.savefig(save, dpi=300)
#    return

def action_sequence_plot(initial_times,
                         action_sequence,
                         harmonic_set,
                         eta_set,
                         max_time,
                         freqs,
                         noise,
                         show=True,
                         save=None):
    """
    PARAMETERS:
    initial_times (list like)
    action_sequence (list)
    harmonic_set (list)
    eta_set (list)
    max_time (float)
    """

    # Define font sizes
    SIZE_TICK_LABEL = 12
    SIZE_AXIS_LABEL = 16
    SIZE_TITLE = 22
    # Define pad sizes
    PAD_AXH = 18
    PAD_DEFAULT_AXIS_LABEL = 14
    PAD_TITLE = 18
    # Define plot line widths
    LW_CURVES = 2
    LW_SEQUENCE = 2.5
    SIZE_SCATTER = 40
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
    mpl.rcParams['axes.titlepad'] = PAD_AXH
    plt.rcParams['figure.constrained_layout.use'] = True
    agent_color = '#984EA3'# Lilac
    noise_color = '#2CA02C'# Green

    fig = plt.figure(layout = 'constrained', figsize=(12, 10))
    #mosaic = """
    #         FPPP11
    #         FPPP22
    #         FPPP33
    #         """
    mosaic = """
             F.P.0
             F.P.1
             F.P.2
             """
    axd = fig.subplot_mosaic(mosaic, width_ratios=[1, 0.1, 3, 0.3, 2])
    # Make harmonic set title
    harmonic_title = 'Harmonic Set $\{'
    abs_h = np.abs(harmonic_set)
    unique_abs_h, counts = np.unique(abs_h, return_counts=True)
    for i, h in np.ndenumerate(unique_abs_h):
        if counts[i] > 1:
            harmonic_title += '\pm' + str(h)
        else:
            harmonic_title += str(h)
        if i[0] < len(unique_abs_h) - 1:
            harmonic_title += ', '
    harmonic_title += '\}$'
    #fig.suptitle('Example Action Sequence with \n' + harmonic_title, fontsize=SIZE_TITLE)

    ##########      PULSE SEQUENCE PLOT      ##########
    ###################################################

    axd['P'].set_title('Example Action Sequence with \n' + harmonic_title, fontsize=SIZE_TITLE)
    # Make axis which plots harmonics used
    axH = axd['P'].twinx()
    # Add gridlines
    axd['P'].grid(axis='y')
    # Set tick label size
    axd['P'].tick_params(axis='both', which='major', labelsize=SIZE_TICK_LABEL) 
    axH.tick_params(axis='y', which='major', labelsize=SIZE_TICK_LABEL)

    # X axis
    axd['P'].set_xlim( (0, max_time) )
    xtick_positions = [0, max_time/4, max_time/2, 3*max_time/4, max_time]
    xtick_labels = ['0', '0.25T', '0.5T', '0.75T', 'T']
    axd['P'].set_xticks(ticks=xtick_positions, labels=xtick_labels)
    axd['P'].set_xlabel('Time', fontsize=SIZE_AXIS_LABEL)

    # Parameters for vlines
    Nstep = len(action_sequence)
    vlines_ymin = np.arange(Nstep + 1) - 0.5
    vlines_ymax = vlines_ymin + 1

    # Lefthand Y axis
    ylim = (-1, Nstep + 1)
    axd['P'].set_ylim(ylim)
    y1tick_locs = vlines_ymin + 0.5
    #y1tick_labels = np.arange(Nstep + 1)
    y1tick_labels = np.flip(np.arange(Nstep + 1))
    axd['P'].set_yticks(ticks=y1tick_locs, labels = y1tick_labels)
    axd['P'].set_ylabel('Steps', fontsize=SIZE_AXIS_LABEL)

    # Righthand Y axis
    axH.set_ylim(ylim)
    #y2tick_locs = y1tick_locs[1:] 
    y2tick_locs = y1tick_locs[:len(y1tick_locs)-1] 
    y2tick_labels = np.flip(np.array([harmonic_set[a] for a in action_sequence]))
    #y2tick_labels = np.array([harmonic_set[a] for a in action_sequence])
    axH.set_yticks(ticks=y2tick_locs, labels = y2tick_labels)
    axH.set_ylabel('Harmonics Chosen', fontsize=SIZE_AXIS_LABEL, rotation=270, labelpad=PAD_TITLE)


    # Plot pulse sequence vertical lines for the initial state and then each subsequent state.
    nStep = len(action_sequence)
    fid_array = np.zeros(nStep + 1)
    ff_slices = np.zeros((3, len(freqs)))
    fid_slices = np.zeros(3)
    step = 0
    pulse_times = np.copy(initial_times)
    axd['P'].vlines(pulse_times, np.flip(vlines_ymin)[step], np.flip(vlines_ymax)[step], color = agent_color, lw=LW_SEQUENCE) # Plot initial state
    #axd['P'].vlines(pulse_times, vlines_ymin[step], vlines_ymax[step], color = agent_color, lw=LW_SEQUENCE) # Plot initial state
    for action in action_sequence:
        ff = ps.FilterFunc(freqs, pulse_times, max_time)
        chi = ps.chi(freqs, noise, ff)
        fid_array[step] = ps.fidelity(chi)
        # Get fidelity and filter function for the first filter function plot
        if step == 0:
            fid_slices[0] = fid_array[step]
            ff_slices[0] = ff
        # Get fidelity and filter function for the second filter function plot
        elif step == int(nStep / 2):
            fid_slices[1] = fid_array[step]
            ff_slices[1] = ff
        step += 1
        pulse_times = kappa(pulse_times, harmonic_set[action], eta_set[action], max_time)
        axd['P'].vlines(pulse_times, np.flip(vlines_ymin)[step], np.flip(vlines_ymax)[step], color = agent_color, lw=LW_SEQUENCE)
    ff = ps.FilterFunc(freqs, pulse_times, max_time)
    chi = ps.chi(freqs, noise, ps.FilterFunc(freqs, pulse_times, max_time))
    fid_array[step] = ps.fidelity(chi)
    # Get fidelity and filter function for the third filter function plot
    ff_slices[2] = ff
    fid_slices[2] = fid_array[step]
    
    ##########         FIDELITY PLOT         ##########
    ###################################################
    
    axd['F'].sharey(axd['P'])
    axd['F'].set_xlabel('Fidelity', fontsize=SIZE_AXIS_LABEL)
    axd['F'].set_ylabel('Steps', fontsize=SIZE_AXIS_LABEL)
    axd['F'].tick_params(axis='both', which='major', labelsize=SIZE_TICK_LABEL) 
    axd['F'].grid(axis='y') # Add gridlines
    axd['F'].scatter(fid_array, np.flip(np.arange(nStep + 1)), s=SIZE_SCATTER)

    ##########         FILTER PLOTS         ##########
    ##################################################

    # Find max index for plotting
    cutoff_idx = len(noise)-1 # Initialize
    for i in range(len(noise)-1, -1, -1):
        if (noise[i] / freqs[i]**2 > 1e-3):
            cutoff_idx = i
            break
    angfreqs = 2 * np.pi * freqs # Convert to angular frequency
    for j in range(3):
        ax_key = str(j)
        axd[ax_key].plot(angfreqs[:cutoff_idx], ff_slices[j, :cutoff_idx], color=agent_color, label='$F(\omega)$', lw=LW_CURVES)
        axd[ax_key].plot(angfreqs[:cutoff_idx], noise[:cutoff_idx], color = noise_color, label='$S(\omega)$', lw=LW_CURVES)
        axd[ax_key].tick_params(axis='both', which='major', labelsize=SIZE_TICK_LABEL) 
        axd[ax_key].set_yscale('log')
        axd[ax_key].legend(prop={'size': SIZE_TICK_LABEL})
        if j == 0:
            axd[ax_key].set_title('Step {0}: Fidelity = {1:.3f}'.format(j, fid_slices[j]), fontsize=SIZE_AXIS_LABEL)
        elif j > 0:
            axd[ax_key].sharex(axd['0'])
            axd[ax_key].sharey(axd['0'])
            if j == 1:
                axd[ax_key].set_title('Step {0}: Fidelity = {1:.3f}'.format(int(nStep/2), fid_slices[j]), fontsize=SIZE_AXIS_LABEL)
            elif j == 2:
                axd[ax_key].set_title('Step {0}: Fidelity = {1:.3f}'.format(nStep, fid_slices[j]), fontsize=SIZE_AXIS_LABEL)
                axd[ax_key].set_xlabel('$\omega$', fontsize=SIZE_AXIS_LABEL)

    if save is not None:
        plt.savefig(save, dpi=300)
    if show is True:
        plt.show()
    return

if __name__=="__main__":

    job_dir = '/home/charlie/Documents/ml/CollectiveAction/eta_scan_data/1_over_f/harmonics_01_02_08/job_00002/run_00001'
    save_dir = '/home/charlie/Documents/ml/CollectiveAction/paper_plots/Fig3.svg'
    action_sequence = list(np.loadtxt(os.path.join(job_dir, 'action.txt'), dtype=int))
    harmonics_arr = np.loadtxt(os.path.join(job_dir, 'harmonics.txt'), dtype=int)
    etaN = np.loadtxt(os.path.join(job_dir, 'etaN.txt'))
    freqs_arr = np.loadtxt(os.path.join(job_dir, 'freq.txt'))
    noise_arr = np.loadtxt(os.path.join(job_dir, 'sOmega.txt'))
    
    max_time = 1
    Npulse = 8
    pulse_times = ps.PDD(Npulse, max_time)
    eta_arr = np.abs(Npulse / harmonics_arr) * etaN

    action_sequence_plot(pulse_times, action_sequence, harmonics_arr, eta_arr, max_time, freqs_arr, noise_arr, save=save_dir, show=False)
    #action_sequence_plot(pulse_times, action_sequence, harmonics_arr, eta_arr, max_time, freqs_arr, noise_arr)

