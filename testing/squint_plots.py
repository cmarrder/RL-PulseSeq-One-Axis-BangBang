import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import os
import pulse_sequence as ps


def crunch_job(max_time,
               freq,
               noise,
               agent_state,
               agent_reward,
               initial_state,
               weights=None):
    """
    PARAMETERS:
    max_time (number): maximum time evolution time
    freq (np.ndarray): frequency mesh
    noise (np.ndarray): noise PSD values for each value of frequency. Curve is the Fermi Dirac distribution.
    agent_state (np.ndarray): best state found by agent. Each element is the time at which pulses are applied.
    agent_filter (np.ndarray): filter function associated with agent_state
    agent_reward (np.ndarray): reward for every learning trial
    weights (Optional, np.ndarray): weights for integration

    RETURNS:
    pulse_timings (np.ndarray, shape (3, Npa)): the times at which the agent applied pulse
    Npa (int): number of pulses the agent actually applied
    Nts (int): number of chances along time mesh at which agent could apply pulse
    filters (np.ndarray, shape (3, len(freq))): the filters as a function of frequency for agent, UDD, and CPMG
    overlaps (np.ndarray, shape (3,)):  Overlap associated with agent's best filter, UDD overlap, and CPMG overlap
    rewards (np.ndarray, shape (3, nTrial)): Reward values for each trial for agent, UDD overlap, and CPMG overlap


    Note: for all of the returned arrays mentioned above, the quantity/quantities associated with
          the agent are in row 0,
          UDD are in row 1,
          CPMG are in row 2.
    """

    # Get pulse timings
    Npa = len(agent_state) # Number pulses applied
    # Put all the timings into an array
    pulse_timings = np.zeros((3, Npa))
    pulse_timings[0] = agent_state
    pulse_timings[1] = ps.UDD(Npa, max_time)
    pulse_timings[2] = ps.CPMG(Npa, max_time)

    # Calculate filters and put them into an array
    filters = np.zeros((3, len(freq)))
    filters[0] = ps.FilterFunc(freq, pulse_timings[0], max_time) # RL Filter 
    filters[1] = ps.FilterFunc(freq, pulse_timings[1], max_time) # UDD filter 
    filters[2] = ps.FilterFunc(freq, pulse_timings[2], max_time) # CPMG filter
    initial_filter = ps.FilterFunc(freq, initial_state, max_time)

    # Calculate overlap functions and put into an array
    overlaps = np.zeros(3)
    overlaps[0] = ps.chi(freq, noise, filters[0], weights=weights) # Overlap associated with agent's best filter
    overlaps[1] = ps.chi(freq, noise, filters[1], weights=weights) # Overlap for UDD
    overlaps[2] = ps.chi(freq, noise, filters[2], weights=weights) # Overlap for CPMG
    initial_chi = ps.chi(freq, noise, initial_filter, weights=weights)


    # Calculate rewards over trial for UDD and CPMG. (Will be constant across trial number).
    # Put these into array with agent_reward
    nTrial = len(agent_reward)
    rewards = np.zeros((3, nTrial))
    rewards[0] = agent_reward
    rewards[1] = ps.RewardFunc(overlaps[1], initial_chi) * np.ones(nTrial) # UDD filter
    rewards[2] = ps.RewardFunc(overlaps[2], initial_chi) * np.ones(nTrial) # CPMG filter
    
    print("overlaps")
    print(overlaps)
    print("initial chi")
    print(initial_chi)

    print("RL reward python")
    print(ps.RewardFunc(overlaps[0], initial_chi))
    print("UDD reward")
    print(ps.RewardFunc(overlaps[1], initial_chi))
    print("CPMG reward")
    print(ps.RewardFunc(overlaps[2], initial_chi))

    print("C++ initial state")
    print(initial_state)
    print("python initial state")
    print(pulse_timings[2])

    return pulse_timings, filters, overlaps, rewards

def plot_job(max_time,
             freq,
             noise,
             pulse_timings,
             Npa,
             Nts,
             filters,
             overlaps,
             rewards,
             agent_loss,
             save = None, show = True, title = None, filterScale = 'linear', noiseScale = 'linear'):
    """
    PARAMETERS:
    max_time (number): maximum time evolution time
    freq (np.ndarray): frequency mesh
    noise (np.ndarray): noise PSD values for each value of frequency. Curve is the Fermi Dirac distribution.
    chem_pot (number): chemical potential associated with the noise.
    temp (number): temperature associated with the noise.
    pulse_timings (np.ndarray, shape (3, Npa)): the times at which the agent applied pulse
    Npa (int): number of pulses the agent actually applied
    Nts (int): number of chances along time mesh at which agent could apply pulse
    filters (np.ndarray, shape (3, len(freq))): the filters as a function of frequency for agent, UDD, and CPMG
    overlaps (np.ndarray, shape (3,)):  Overlap associated with agent's best filter, UDD overlap, and CPMG overlap
    rewards (np.ndarray, shape (3, nTrial)): Reward values for each trial for agent, UDD overlap, and CPMG overlap
    agent_loss (np.ndarray): loss for every learning trial
    save (Optional, string): indicates whether or not to save. If string is provided, figure is saved to the filename represented by string.
    show (Optional, bool): indicated whether to show figure using matplotlib's interactive environment
   
    Note: for all of the arrays mentioned above, the quantity/quantities associated with
          the agent are in row 0,
          UDD are in row 1,
          CPMG are in row 2.
          Also, save and show cannot be true at the same time.

    RETURNS:
    None
    """
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['xtick.major.size'] = 6
    mpl.rcParams['xtick.major.width'] = 4 
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['xtick.minor.width'] = 3
    mpl.rcParams['ytick.major.size'] = 6
    mpl.rcParams['ytick.major.width'] = 4 
    mpl.rcParams['ytick.minor.size'] = 3
    mpl.rcParams['ytick.minor.width'] = 3
    mpl.rcParams['axes.titlepad'] = 10 
    # Define font sizes
    SIZE_DEFAULT = 18
    SIZE_LARGE = 20
    # Define pad
    PAD_DEFAULT = 10

    #fig = plt.figure(layout = 'constrained', figsize = (16, 8))
    fig = plt.figure(layout = 'constrained', figsize = (12, 8))
    mosaic = """
             BE
             AD
             CF
             """
    axd = fig.subplot_mosaic(mosaic, width_ratios=[1, 2])
    
    UDDColor = '#377EB8'# Curious Blue
    UDDLinestyle = 'solid'
    UDDLabel = r'$\chi_{UDD} = $' + '{:.3e}'.format(overlaps[1])

    agentColor = '#984EA3'# Deep Lilac
    agentLinestyle = 'solid'
    agentLabel = r'$\chi_{RL} = $' + '{:.3e}'.format(overlaps[0])
    
    CPMGColor = '#FF7F00'# Dark Orange
    CPMGLinestyle = 'solid'
    CPMGLabel = r'$\chi_{CPMG} = $' + '{:.3e}'.format(overlaps[2])


    freq *= 2*np.pi # Make angular frequency
    filters /= np.pi

    filter_y_Label = r'$F(\omega)$'
    filter_x_Label = r'$\omega$'
    filter_UDD_Label = r'$F(\omega)_{UDD}$'
    filter_RL_Label = r'$F(\omega)_{RL}$'
    filter_CPMG_Label = r'$F(\omega)_{CPMG}$'
    
    noiseColor = '#2CA02C'#'#4DAF4A'# Fruit Salad, green
    noiseLinestyle = 'dashdot'
    noiseLabel = r'$S(\omega)$'
    """
    filter_y_Label = r'$F(\nu)$'
    filter_x_Label = r'$\nu$ [1/time]'
    filter_UDD_Label = r'$F(\nu)_{UDD}$'
    filter_RL_Label = r'$F(\nu)_{RL}$'
    filter_CPMG_Label = r'$F(\nu)_{CPMG}$'
    noiseColor = '#2CA02C'#'#4DAF4A'# Fruit Salad, green
    noiseLinestyle = 'dashdot'
    noiseLabel = r'$S(\nu)$'
    """

    lwps = 6 # Linewidth for pulse sequence plots
    lwspectrum = 5 # Linewidth for filter/noise plots
    # Find max index for plotting
    cutoffIdx = len(noise)-1 # Initialize
    for i in range(len(noise)-1, -1, -1):
        #if (noise[i] / freq[i]**2 > 1e-6):
        if (noise[i] / freq[i]**2 > 1e-4):
            cutoffIdx = i
            break
    print("cutoffFreq = {}".format(freq[cutoffIdx]))


    if title is not None:
        fig.suptitle(title, fontsize = 1.5*SIZE_LARGE)
    
    # Plot pulse sequence over time

    axd['A'].set_title('UDD ($N_{{pulse}} = {}$)'.format(int(Npa)), fontsize = SIZE_DEFAULT)
    axd['A'].sharex(axd['C'])
    axd['A'].vlines(pulse_timings[1], 0, 1, color=UDDColor, linestyle=UDDLinestyle, linewidth = lwps)
    axd['A'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['A'].tick_params(labelbottom=False) # Remove x axis labels
    axd['A'].grid(axis = 'x')

    axd['B'].set_title('RL ($N_{{pulse}} = {}$)'.format(int(Npa)), fontsize = SIZE_DEFAULT)
    axd['B'].sharex(axd['C'])
    axd['B'].vlines(pulse_timings[0], 0, 1, color=agentColor, linewidth = lwps)
    axd['B'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['B'].tick_params(labelbottom=False) # Remove x axis labels
    axd['B'].grid(axis = 'x')
   
    axd['C'].set_title('CPMG ($N_{{pulse}} = {}$)'.format(int(Npa)), fontsize = SIZE_DEFAULT)
    axd['C'].set_xlabel('Time', fontsize=SIZE_LARGE)
    axd['C'].set_xlim(0, max_time)
    axd['C'].vlines(pulse_timings[2], 0, 1, color=CPMGColor, linestyle=CPMGLinestyle, linewidth=lwps)
    axd['C'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['C'].grid(axis = 'x')
    axd['C'].tick_params(axis='both', which='major', labelsize=SIZE_DEFAULT)
    axd['C'].tick_params(axis='both', which='minor', labelsize=SIZE_DEFAULT)


    # Plot filter functions and noise PSD

    axd['D'].set_ylabel(filter_y_Label, fontsize=SIZE_LARGE, labelpad=PAD_DEFAULT)
    axd['D'].sharey(axd['F'])
    axUDDNoise = axd['D'].twinx()
    #axUDDNoise.yaxis.set_major_locator(ticker.MaxNLocator(5)) # Set number y ticks
    axUDDNoise.set_ylabel(noiseLabel, fontsize=SIZE_LARGE, labelpad=PAD_DEFAULT)
    filterPlotUDD = axd['D'].plot(freq[:cutoffIdx], filters[1][:cutoffIdx], color=UDDColor, linestyle=UDDLinestyle, label = filter_UDD_Label, linewidth=lwspectrum)
    noisePlotUDD = axUDDNoise.plot(freq[:cutoffIdx], noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = noiseLabel, linewidth=lwspectrum)
    linesUDD = filterPlotUDD + noisePlotUDD
    labelsUDD = [l.get_label() for l in linesUDD]
    axd['D'].tick_params(labelbottom=False) # Remove x axis labels
    axd['D'].tick_params(axis='y', which='major', labelsize=SIZE_DEFAULT)
    axd['D'].set_title(UDDLabel, fontsize=SIZE_LARGE)
    axd['D'].legend(linesUDD, labelsUDD, fontsize=SIZE_LARGE)
    axUDDNoise.tick_params(axis='y', which='major', labelsize=SIZE_DEFAULT)


    axd['E'].set_ylabel(filter_y_Label, fontsize=SIZE_LARGE, labelpad=PAD_DEFAULT)
    axd['E'].sharey(axd['F'])
    axRLNoise = axd['E'].twinx()
    axRLNoise.set_ylabel(noiseLabel, fontsize=SIZE_LARGE, labelpad=PAD_DEFAULT)
    filterPlotRL = axd['E'].plot(freq[:cutoffIdx], filters[0][:cutoffIdx], color=agentColor, linestyle=agentLinestyle, label = filter_RL_Label, linewidth=lwspectrum)
    noisePlotRL = axRLNoise.plot(freq[:cutoffIdx], noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = noiseLabel, linewidth=lwspectrum)
    linesRL = filterPlotRL + noisePlotRL
    labelsRL = [l.get_label() for l in linesRL]
    axd['E'].tick_params(labelbottom=False) # Remove x axis labels
    axd['E'].tick_params(axis='y', which='major', labelsize=SIZE_DEFAULT)
    axd['E'].set_title(agentLabel, fontsize=SIZE_LARGE)
    axd['E'].legend(linesRL, labelsRL, fontsize=SIZE_LARGE)
    axRLNoise.tick_params(axis='y', which='major', labelsize=SIZE_DEFAULT)

    axd['F'].set_ylabel(filter_y_Label, fontsize=SIZE_LARGE, labelpad=PAD_DEFAULT)
    #axd['F'].yaxis.set_major_locator(ticker.MaxNLocator(5)) # Set number y ticks
    axCPMGNoise = axd['F'].twinx()
    axCPMGNoise.set_ylabel(noiseLabel, fontsize=SIZE_LARGE, labelpad=PAD_DEFAULT)
    filterPlotCPMG = axd['F'].plot(freq[:cutoffIdx], filters[2][:cutoffIdx], color=CPMGColor, linestyle=CPMGLinestyle, label = filter_CPMG_Label, linewidth=lwspectrum)
    noisePlotCPMG = axCPMGNoise.plot(freq[:cutoffIdx], noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = noiseLabel, linewidth=lwspectrum)
    linesCPMG = filterPlotCPMG + noisePlotCPMG
    labelsCPMG = [l.get_label() for l in linesCPMG]
    axd['F'].tick_params(axis='both', which='major', labelsize=SIZE_DEFAULT)
    axd['F'].set_title(CPMGLabel, fontsize=SIZE_LARGE)
    axd['F'].set_xlabel(filter_x_Label, fontsize=SIZE_LARGE)
    axd['F'].legend(linesCPMG, labelsCPMG, fontsize=SIZE_LARGE)
    axd['F'].set_yscale(filterScale)
    axCPMGNoise.tick_params(axis='y', which='major', labelsize=SIZE_DEFAULT)

    axUDDNoise.sharey(axCPMGNoise)
    axRLNoise.sharey(axCPMGNoise)
    axCPMGNoise.set_yscale(noiseScale)

    #fig.tight_layout()
    if show:
        plt.show()
    elif save is not None:
        plt.savefig(save, dpi=300)
        plt.close()

    return


if __name__=='__main__':
     
    # Load data
    #oDir = '/home/charlie/Documents/ml/CollectiveAction/data/job_00000'
    oDir = '/home/charlie/Documents/ml/CollectiveAction/data_1_over_f/job_00000'

    nPulse = int( np.loadtxt(os.path.join(oDir, 'nPulse.txt')) ) # Number of pulse applications
    nTimeStep = int( np.loadtxt(os.path.join(oDir, 'nTimeStep.txt')) )# Number of pulse chances/locations
    tMax = np.loadtxt(os.path.join(oDir, 'maxTime.txt'))
    noiseParam1 = np.loadtxt(os.path.join(oDir, 'noiseParam1.txt'))
    noiseParam2 = np.loadtxt(os.path.join(oDir, 'noiseParam2.txt'))

    freq = np.loadtxt(os.path.join(oDir, 'freq.txt'))
    sOmega = np.loadtxt(os.path.join(oDir, 'sOmega.txt'))
    
    finalState = np.loadtxt(os.path.join(oDir, 'state.txt'))
    reward = np.loadtxt(os.path.join(oDir, 'reward.txt'))
    loss = np.loadtxt(os.path.join(oDir, 'loss.txt'))
    initialState = np.loadtxt(os.path.join(oDir, 'initialState.txt'))

    print('finalState')
    print(finalState)

   
    pulse_timings, filters, overlaps, rewards = crunch_job(tMax,
                                                          freq,
                                                          sOmega,
                                                          finalState,
                                                          reward,
                                                          initialState)

    #job_title = '$\mu = {0}, T = {1}$'.format(noiseParam1, noiseParam2)
    #job_title = '$bandCenter = {0}*cpmgPeak, bandWidth = {1} / maxTime$'.format(noiseParam2, noiseParam1)
    #job_title = 'Sum of Lorentzians on CPMG Peaks'
    #job_title = 'Double Lorentzian Noise'
    job_title = '1 / f Noise'
    save_job = 'squint_plots/1_over_f.png' 
    #save_job = 'squint_plots/doubleLorentzian.png' 
    show_job = False

    plot_job(tMax,
             freq,
             sOmega,
             pulse_timings,
             nPulse,
             nTimeStep,
             filters,
             overlaps,
             rewards,
             loss,
             save = save_job, show = show_job, title = job_title,
             filterScale = 'linear', noiseScale = 'linear')

    """
    dd = '/home/charlie/Documents/ml/CollectiveAction/data'
    pd = '/home/charlie/Documents/ml/CollectiveAction/plots'
    temperature_sweep(dd, pd, show=False)
    """ 
