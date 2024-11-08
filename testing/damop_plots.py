import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pulse_sequence as ps


def crunch_job(max_time,
               freq,
               noise,
               agent_state,
               agent_filter,
               agent_reward,
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
    Npc (int): number of chances along time mesh at which agent could apply pulse
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
    filters[0] = agent_filter 
    filters[1] = ps.FilterFunc(freq, pulse_timings[1], max_time) # UDD filter 
    filters[2] = ps.FilterFunc(freq, pulse_timings[2], max_time) # CPMG filter

    # Calculate overlap functions and put into an array
    overlaps = np.zeros(3)
    overlaps[0] = ps.chi(freq, noise, agent_filter, weights=weights) # Overlap associated with agent's best filter
    overlaps[1] = ps.chi(freq, noise, filters[1], weights=weights) # Overlap for UDD
    overlaps[2] = ps.chi(freq, noise, filters[2], weights=weights) # Overlap for CPMG

    average_chi = 0#ps.chi_avg(freq, noise, max_time, Npc, weights=weights)
   
    # Calculate rewards over trial for UDD and CPMG. (Will be constant across trial number).
    # Put these into array with agent_reward
    nTrial = len(agent_reward)
    rewards = np.zeros((3, nTrial))
    rewards[0] = agent_reward
    rewards[1] = ps.RewardFunc(overlaps[1], average_chi) * np.ones(nTrial) # UDD filter
    rewards[2] = ps.RewardFunc(overlaps[2], average_chi) * np.ones(nTrial) # CPMG filter

    return pulse_timings, filters, overlaps, rewards

def plot_job(max_time,
             freq,
             noise,
             chem_pot,
             temp,
             pulse_timings,
             Npa,
             Npc,
             filters,
             overlaps,
             rewards,
             agent_loss,
             save = None, show = False):
    """
    PARAMETERS:
    max_time (number): maximum time evolution time
    freq (np.ndarray): frequency mesh
    noise (np.ndarray): noise PSD values for each value of frequency. Curve is the Fermi Dirac distribution.
    chem_pot (number): chemical potential associated with the noise.
    temp (number): temperature associated with the noise.
    pulse_timings (np.ndarray, shape (3, Npa)): the times at which the agent applied pulse
    Npa (int): number of pulses the agent actually applied
    Npc (int): number of chances along time mesh at which agent could apply pulse
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

    RETURNS:
    None
    """

    fig = plt.figure(layout = 'constrained', figsize = (16, 8))
    
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 2 
    mpl.rcParams['xtick.minor.size'] = 1
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 2 
    mpl.rcParams['ytick.minor.size'] = 1
    mpl.rcParams['ytick.minor.width'] = 1

    mosaic = """
             DEF
             ABC
             """
    axd = fig.subplot_mosaic(mosaic, width_ratios=[1, 1, 1], height_ratios=[6,1])
    
    UDDColor = '#377EB8'# Curious Blue
    UDDLinestyle = 'solid'
    UDDLabel = r'$\chi_{UDD} = $' + '{:.3e}'.format(overlaps[1])

    agentColor = '#984EA3'# Deep Lilac
    agentLinestyle = 'solid'
    agentLabel = r'$\chi_{Agent} = $' + '{:.3e}'.format(overlaps[0])
    
    CPMGColor = '#FF7F00'# Dark Orange
    CPMGLinestyle = 'solid'
    CPMGLabel = r'$\chi_{CPMG} = $' + '{:.3e}'.format(overlaps[2])

    noiseColor = '#4DAF4A'# Fruit Salad, green
    noiseLinestyle = 'dashdot'
    noiseLabel = r'$S(\nu)$'

    lwps = 4 # Linewidth for pulse sequence plots
    lwspectrum = 2.5 # Linewidth for filter/noise plots
    maxIdx = int(0.5 * len(noise))
   
    # Define font sizes
    SIZE_DEFAULT = 14
    SIZE_LARGE = 16


    # ONLY NEED THIS FOR FERMI-DIRAC
    fig.suptitle('$\mu = {0}, T = {1}$'.format(chem_pot, temp), fontsize = SIZE_LARGE)
    
    # Plot pulse sequence over time

    axd['A'].set_title('UDD ($N_{{pulse}} = {}$)'.format(int(Npa)), fontsize = SIZE_LARGE)
    axd['A'].set_xlabel('Time', fontsize = SIZE_LARGE)
    axd['A'].sharex(axd['C'])
    axd['A'].vlines(pulse_timings[1], 0, 1, color=UDDColor, linestyle=UDDLinestyle, linewidth = lwps)
    axd['A'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['A'].tick_params(axis='both', which='major', labelsize=SIZE_LARGE)
    axd['A'].tick_params(axis='both', which='minor', labelsize=SIZE_DEFAULT)
   
    axd['B'].set_title('Agent ($N_{{chances}} = {0}, N_{{pulse}} = {1}$)'.format(int(Npc), int(Npa)), fontsize = SIZE_LARGE)
    axd['B'].sharex(axd['C'])
    axd['B'].set_xlabel('Time', fontsize = SIZE_LARGE)
    axd['B'].vlines(pulse_timings[0], 0, 1, color=agentColor, linewidth = lwps)
    axd['B'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['B'].tick_params(axis='both', which='major', labelsize=SIZE_LARGE)
    axd['B'].tick_params(axis='both', which='minor', labelsize=SIZE_DEFAULT)
   
    axd['C'].set_title('CPMG ($N_{{pulse}} = {}$)'.format(int(Npa)), fontsize = SIZE_LARGE)
    axd['C'].set_xlabel('Time', fontsize = SIZE_LARGE)
    axd['C'].set_xlim(0, max_time)
    axd['C'].vlines(pulse_timings[2], 0, 1, color=CPMGColor, linestyle=CPMGLinestyle, linewidth=lwps)
    axd['C'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['C'].tick_params(axis='both', which='major', labelsize=SIZE_LARGE)
    axd['C'].tick_params(axis='both', which='minor', labelsize=SIZE_DEFAULT)

    # Plot filter functions and noise PSD
   
    #axd['D'].sharex(axd['F'])
    axd['D'].set_ylabel(r'$F(\nu)$', fontsize = SIZE_LARGE)
    axUDDNoise = axd['D'].twinx()
    axUDDNoise.set_ylabel(noiseLabel, fontsize = SIZE_LARGE)
    filterPlotUDD = axd['D'].plot(freq[:maxIdx], filters[1][:maxIdx], color=UDDColor, linestyle=UDDLinestyle, label = r'$F(\nu)_{UDD}$', linewidth = lwspectrum)
    noisePlotUDD = axUDDNoise.plot(freq[:maxIdx], noise[:maxIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\nu)$', linewidth = lwspectrum)
    linesUDD = filterPlotUDD + noisePlotUDD
    labelsUDD = [l.get_label() for l in linesUDD]
    axd['D'].set_title(UDDLabel, fontsize = SIZE_LARGE)
    axd['D'].legend(linesUDD, labelsUDD, fontsize = SIZE_LARGE)
    axd['D'].set_xlabel(r'$\nu$ [1/time]', fontsize = SIZE_LARGE)
    axd['D'].tick_params(axis='both', which='major', labelsize=SIZE_LARGE)
    axd['D'].tick_params(axis='both', which='minor', labelsize=SIZE_DEFAULT)
    axUDDNoise.tick_params(axis='y', which='major', labelsize=SIZE_LARGE)
    axUDDNoise.tick_params(axis='y', which='minor', labelsize=SIZE_DEFAULT)


    axd['E'].set_ylabel(r'$F(\nu)$', fontsize = SIZE_LARGE)
    axAgentNoise = axd['E'].twinx()
    axAgentNoise.set_ylabel(noiseLabel, fontsize = SIZE_LARGE)
    filterPlotAgent = axd['E'].plot(freq[:maxIdx], filters[0][:maxIdx], color=agentColor, linestyle=agentLinestyle, label = r'$F(\nu)_{Agent}$', linewidth = lwspectrum)
    noisePlotAgent = axAgentNoise.plot(freq[:maxIdx], noise[:maxIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\nu)$', linewidth = lwspectrum)
    linesAgent = filterPlotAgent + noisePlotAgent
    labelsAgent = [l.get_label() for l in linesAgent]
    axd['E'].set_title(agentLabel, fontsize = SIZE_LARGE)
    axd['E'].legend(linesAgent, labelsAgent, fontsize = SIZE_LARGE)
    axd['E'].set_xlabel(r'$\nu$ [1/time]', fontsize = SIZE_LARGE)
    axd['E'].tick_params(axis='both', which='major', labelsize=SIZE_LARGE)
    axd['E'].tick_params(axis='both', which='minor', labelsize=SIZE_DEFAULT)
    axAgentNoise.tick_params(axis='y', which='major', labelsize=SIZE_LARGE)
    axAgentNoise.tick_params(axis='y', which='minor', labelsize=SIZE_DEFAULT)

    axd['F'].set_ylabel(r'$F(\nu)$', fontsize = SIZE_LARGE)
    axCPMGNoise = axd['F'].twinx()
    axCPMGNoise.set_ylabel(noiseLabel, fontsize = SIZE_LARGE)
    filterPlotCPMG = axd['F'].plot(freq[:maxIdx], filters[2][:maxIdx], color=CPMGColor, linestyle=CPMGLinestyle, label = r'$F(\nu)_{CPMG}$', linewidth = lwspectrum)
    noisePlotCPMG = axCPMGNoise.plot(freq[:maxIdx], noise[:maxIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\nu)$', linewidth = lwspectrum)
    linesCPMG = filterPlotCPMG + noisePlotCPMG
    labelsCPMG = [l.get_label() for l in linesCPMG]
    axd['F'].set_title(CPMGLabel, fontsize = SIZE_LARGE)
    axd['F'].set_xlabel(r'$\nu$ [1/time]', fontsize = SIZE_LARGE)
    axd['F'].legend(linesCPMG, labelsCPMG, fontsize = SIZE_LARGE)
    axd['F'].tick_params(axis='both', which='major', labelsize=SIZE_LARGE)
    axd['F'].tick_params(axis='both', which='minor', labelsize=SIZE_DEFAULT)
    axCPMGNoise.tick_params(axis='y', which='major', labelsize=SIZE_LARGE)
    axCPMGNoise.tick_params(axis='y', which='minor', labelsize=SIZE_DEFAULT)


    #fig.tight_layout()
    if show:
        plt.show()
    elif save:
        plt.savefig(save)
        plt.close(fig=fig)

    return

def temperature_sweep(data_dir, plot_dir, subdir_prefix='job', show = True):
    """
    PARAMETERS:
    data_dir (string): directory containing the subdirectories which contain the data files
    plot_dir (string): directory where plots will be saved
    subdir_prefix (Optional, string): the prefix of all the subdirectories
    show (bool): indicates whether to show the temperature plot using matplotlib's interactive environment

    RETURNS:
    None
    """
    # List will store temperature values
    temperatures = []

    # Lists will store rewards over temperature
    max_rewards = []
    UDD_rewards = []
    CPMG_rewards = []

    # List will store overlap values over temperature
    max_chi = []
    UDD_chi = []
    CPMG_chi = []

    temp_min = 1e300
    temp_max = 0

    S_temp_min = 0
    S_temp_max = 0
    counter = 0

    print("Reading from directory "+ data_dir)
    for file in os.listdir(data_dir):
        if file.startswith(subdir_prefix):

            try:
                oDir = os.path.join(data_dir, file)

                # Load data

                initials = np.loadtxt(os.path.join(oDir, 'initials.txt'))
                freq = np.loadtxt(os.path.join(oDir, 'freq.txt'))
                sModSq = np.loadtxt(os.path.join(oDir, 'sOmegaAbs2.txt'))
                #weights = np.loadtxt(os.path.join(oDir, 'weights.txt'))
                weights = None
                #sModSq = ps.fermi_dirac(freq, chemPotential, temperature)
                agentFilter = np.loadtxt(os.path.join(oDir, 'fOmegaAbs2.txt'))
                finalState = np.loadtxt(os.path.join(oDir, 'state.txt'))
                reward = np.loadtxt(os.path.join(oDir, 'reward.txt'))
                loss = np.loadtxt(os.path.join(oDir, 'loss.txt'))
                
                Npa = initials[0] # Number of pulse applications
                Npc = initials[1] # Number of pulse chances/locations
                tMax = initials[2]
                chemPotential = initials[5]
                temperature = initials[6]


                # Get min and max temperatures
                if temperature < temp_min:
                    temp_min = temperature
                    S_temp_min = sModSq
                if temperature > temp_max:
                    temp_max = temperature
                    S_temp_max = sModSq

                pulse_timings, filters, overlaps, rewards = crunch_job(tMax,
                                                                      freq,
                                                                      sModSq,
                                                                      finalState,
                                                                      agentFilter,
                                                                      reward,
                                                                      weights=weights)
                plot_job(tMax,
                         freq,
                         sModSq,
                         chemPotential,
                         temperature,
                         pulse_timings,
                         Npa,
                         Npc,
                         filters,
                         overlaps,
                         rewards,
                         loss,
                         save = os.path.join(plot_dir, file),
                         show = False)


                # Append temperatures, rewards, and overlaps to lists
                temperatures.append(temperature)

                max_rewards.append(np.max(rewards[0]))
                UDD_rewards.append(rewards[1,0])
                CPMG_rewards.append(rewards[2,0])

                max_chi.append(overlaps[0])
                UDD_chi.append(overlaps[1])
                CPMG_chi.append(overlaps[2])
                
                print(counter)

                counter += 1
            
            except FileNotFoundError:
                continue

    print('Done with individual plots')

    # Plot chi and reward over temperature
    fig, ax = plt.subplots(layout = 'constrained', figsize = (16, 8))
    
    UDDColor = '#377EB8'# Curious Blue
    UDDLinestyle = 'solid'
    UDDLabel = 'UDD'
    UDDMarker = 'x'

    agentColor = '#984EA3'# Deep Lilac
    agentLinestyle = 'solid'
    agentLabel = 'Agent' 
    agentMarker = '2'
    
    CPMGColor = '#FF7F00'# Dark Orange
    CPMGLinestyle = 'solid'
    CPMGLabel = 'CPMG'
    CPMGMarker = '+'

    noiseColor = '#4DAF4A'# Fruit Salad, green
    noiseLinestyle = 'dashdot'
    noiseLabel = r'$|S(\nu)|^2$'

    markerSize = 100
    markerLW = 2 # Linewidth for pulse sequence plots
    
    # Define font sizes
    SIZE_DEFAULT = 14
    SIZE_LARGE = 16
    
    # Plot overlaps. Use scatter since temperatures might be out of order due to file reading.
#    ax.scatter(temperatures, max_chi, color = agentColor, label = agentLabel, marker = agentMarker, s = markerSize, linewidth = markerLW)
#    ax.scatter(temperatures, UDD_chi, color = UDDColor, label = UDDLabel, marker = UDDMarker, s = markerSize, linewidth = markerLW)
#    ax.scatter(temperatures, CPMG_chi, color = CPMGColor, label = CPMGLabel, marker = CPMGMarker, s = markerSize, linewidth = markerLW)
    ax.plot(temperatures, max_chi, color = agentColor, label = agentLabel, linewidth = markerLW)
    ax.plot(temperatures, UDD_chi, color = UDDColor, label = UDDLabel, linewidth = markerLW)
    ax.plot(temperatures, CPMG_chi, color = CPMGColor, label = CPMGLabel, linewidth = markerLW)
    ax.set_xlabel('Temperature', fontsize = SIZE_LARGE)
    ax.set_ylabel('$\chi$', fontsize = SIZE_LARGE)
    ax.legend(fontsize = SIZE_LARGE)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=SIZE_LARGE)
    ax.tick_params(axis='both', which='minor', labelsize=SIZE_DEFAULT)

    
    plt.savefig(os.path.join(plot_dir, 'temperature.png'))
    if show:
        plt.show()

    print('done')
    return



if __name__=='__main__':
    
    # Load data

    #oDir = '/home/charlie/Murray/Filter2/test_temp/test_00004'
    oDir = '/home/charlie/Documents/ml/Filter/temp_scan_data/job_00000'


#    initials = np.loadtxt(os.path.join(oDir, 'initials.txt'))
#    Npa = initials[0] # Number of pulse applications
#    Npc = initials[1] # Number of pulse chances/locations
#    tMax = initials[2]
#    chemPotential = initials[5]
#    temperature = initials[6]
#
#    freq = np.loadtxt(os.path.join(oDir, 'freq.txt'))
#    sModSq = np.loadtxt(os.path.join(oDir, 'sOmegaAbs2.txt'))
#    agentFilter = np.loadtxt(os.path.join(oDir, 'fOmegaAbs2.txt'))
#    
#    finalState = np.loadtxt(os.path.join(oDir, 'state.txt'))
#    reward = np.loadtxt(os.path.join(oDir, 'reward.txt'))
#    loss = np.loadtxt(os.path.join(oDir, 'loss.txt'))
#
#    save = False # 'tplot.png'
#    show = True
#   
#    pulse_timings, filters, overlaps, rewards = crunch_job(tMax,
#                                                          freq,
#                                                          sModSq,
#                                                          finalState,
#                                                          agentFilter,
#                                                          reward)
#    plot_job(tMax,
#             freq,
#             sModSq,
#             chemPotential,
#             temperature,
#             pulse_timings,
#             Npa,
#             Npc,
#             filters,
#             overlaps,
#             rewards,
#             loss,
#             save = None, show = True)

    dd = '/home/charlie/Documents/ml/Filter/temp_scan_data'
    pd = '/home/charlie/Documents/ml/Filter/temp_scan_plots'
    temperature_sweep(dd, pd, show=True)
