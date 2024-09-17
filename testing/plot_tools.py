import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pulse_sequence as ps


def crunch_job(max_time,
               freq,
               noise,
               agent_state,
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
    filters[0] = ps.FilterFunc(freq, pulse_timings[0], max_time) # Agent Filter 
    filters[1] = ps.FilterFunc(freq, pulse_timings[1], max_time) # UDD filter 
    filters[2] = ps.FilterFunc(freq, pulse_timings[2], max_time) # CPMG filter

    # Calculate overlap functions and put into an array
    overlaps = np.zeros(3)
    overlaps[0] = ps.chi(freq, noise, filters[0], weights=weights) # Overlap associated with agent's best filter
    overlaps[1] = ps.chi(freq, noise, filters[1], weights=weights) # Overlap for UDD
    overlaps[2] = ps.chi(freq, noise, filters[2], weights=weights) # Overlap for CPMG

    # Calculate rewards over trial for UDD and CPMG. (Will be constant across trial number).
    # Put these into array with agent_reward
    nTrial = len(agent_reward)
    rewards = np.zeros((3, nTrial))
    rewards[0] = agent_reward
    rewards[1] = ps.RewardFunc(overlaps[1]) * np.ones(nTrial) # UDD filter
    rewards[2] = ps.RewardFunc(overlaps[2]) * np.ones(nTrial) # CPMG filter
    
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

    RETURNS:
    None
    """

    fig = plt.figure(layout = 'constrained', figsize = (16, 8))
    mosaic = """
             ADG
             BEH
             CF.
             """
    axd = fig.subplot_mosaic(mosaic, width_ratios=[1, 2, 2])
    
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

    lwps = 2 # Linewidth for pulse sequence plots
    maxIdx = int(0.5 * len(noise))
    
    if title is not None:
        fig.suptitle(title)
    
    # Plot pulse sequence over time

    axd['A'].set_title('UDD ($N_{{pulse}} = {}$)'.format(int(Npa)))
    axd['A'].sharex(axd['C'])
    axd['A'].vlines(pulse_timings[1], 0, 1, color=UDDColor, linestyle=UDDLinestyle, linewidth = lwps)
    axd['A'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['A'].tick_params(labelbottom=False) # Remove x axis labels
   
    axd['B'].set_title('Agent ($N_{{time step}} = {0}, N_{{pulse}} = {1}$)'.format(int(Nts), int(Npa)))
    axd['B'].sharex(axd['C'])
    axd['B'].vlines(pulse_timings[0], 0, 1, color=agentColor, linewidth = lwps)
    axd['B'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['B'].tick_params(labelbottom=False) # Remove x axis labels
   
    axd['C'].set_title('CPMG ($N_{{pulse}} = {}$)'.format(int(Npa)))
    axd['C'].set_xlabel('Time')
    axd['C'].set_xlim(0, max_time)
    axd['C'].vlines(pulse_timings[2], 0, 1, color=CPMGColor, linestyle=CPMGLinestyle, linewidth=lwps)
    axd['C'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks

    # Plot filter functions and noise PSD
   
    #axd['D'].sharex(axd['F'])
    axd['D'].set_ylabel(r'$F(\nu)$')
    axUDDNoise = axd['D'].twinx()
    axUDDNoise.set_ylabel(noiseLabel)
    filterPlotUDD = axd['D'].plot(freq[:maxIdx], filters[1][:maxIdx], color=UDDColor, linestyle=UDDLinestyle, label = r'$F(\nu)_{UDD}$')
    #axd['D'].plot(freq, noise, color=noiseColor, linestyle=noiseLinestyle, label = r'$|S(\nu)|^2$')
    noisePlotUDD = axUDDNoise.plot(freq[:maxIdx], noise[:maxIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\nu)$')
    linesUDD = filterPlotUDD + noisePlotUDD
    labelsUDD = [l.get_label() for l in linesUDD]
    axd['D'].tick_params(labelbottom=False) # Remove x axis labels
    #axUDDNoise.set_ylim(top = maxNoise + 0.1)
    #axUDDNoise.relim()
    axd['D'].set_title(UDDLabel)
    axd['D'].legend(linesUDD, labelsUDD)


    #axd['E'].sharex(axd['F'])
    axd['E'].set_ylabel(r'$F(\nu)$')
    axAgentNoise = axd['E'].twinx()
    axAgentNoise.set_ylabel(noiseLabel)
    filterPlotAgent = axd['E'].plot(freq[:maxIdx], filters[0][:maxIdx], color=agentColor, linestyle=agentLinestyle, label = r'$F(\nu)_{Agent}$')
    noisePlotAgent = axAgentNoise.plot(freq[:maxIdx], noise[:maxIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\nu)$')
    linesAgent = filterPlotAgent + noisePlotAgent
    labelsAgent = [l.get_label() for l in linesAgent]
    axd['E'].tick_params(labelbottom=False) # Remove x axis labels
    axd['E'].set_title(agentLabel)
    axd['E'].legend(linesAgent, labelsAgent)
    # ONLY NEED THIS FOR FERMI-DIRAC
    #axd['E'].set_title('$\mu = {0}, T = {1}$'.format(chem_pot, temp))

    axd['F'].set_ylabel(r'$F(\nu)$')
    axCPMGNoise = axd['F'].twinx()
    axCPMGNoise.set_ylabel(noiseLabel)
    filterPlotCPMG = axd['F'].plot(freq[:maxIdx], filters[2][:maxIdx], color=CPMGColor, linestyle=CPMGLinestyle, label = r'$F(\nu)_{CPMG}$')
    noisePlotCPMG = axCPMGNoise.plot(freq[:maxIdx], noise[:maxIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\nu)$')
    linesCPMG = filterPlotCPMG + noisePlotCPMG
    labelsCPMG = [l.get_label() for l in linesCPMG]
    axd['F'].set_title(CPMGLabel)
    axd['F'].set_xlabel(r'$\nu$ [1/time]')
    axd['F'].legend(linesCPMG, labelsCPMG)
    # ONLY NEED THIS FOR FERMI-DIRAC
    #axd['E'].set_title('$\mu = {0}, T = {1}$'.format(chem_pot, temp))

    # Plot reward and loss over trials
    
    nTrial = rewards.shape[1]
    trials = np.arange(1, nTrial + 1) # Do arange(1, n+1) so that index starts at 1, ends at nTrial
    axd['G'].set_ylabel('Reward')
    axd['G'].sharex(axd['H'])
    axd['G'].plot(trials, rewards[0], color=agentColor, label = 'Agent') 
    axd['G'].plot(trials, rewards[1], color=UDDColor, linestyle=UDDLinestyle, label = 'UDD') 
    axd['G'].plot(trials, rewards[2], color=CPMGColor, linestyle=CPMGLinestyle, label = 'CPMG')
    axd['G'].set_yscale('log')

    # Get index of maximum rewards
    idxMax = np.argmax(rewards[0])
    axd['G'].scatter(trials[idxMax], rewards[0][idxMax], color=agentColor, label = 'Agent Max Reward')
    axd['G'].legend()

    axd['H'].set_ylabel('Average Loss')
    axd['H'].set_xlabel('Trials')
    axd['H'].plot(trials, agent_loss, color=agentColor)

    #fig.tight_layout()
    if show:
        plt.show()
    elif save:
        plt.savefig(save)
        plt.close()

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

                freq = np.loadtxt(os.path.join(oDir, 'freq.txt'))
                sOmega = np.loadtxt(os.path.join(oDir, 'sOmega.txt'))
                #weights = np.loadtxt(os.path.join(oDir, 'weights.txt'))
                weights = None
                #sOmega = ps.fermi_dirac(freq, chemPotential, temperature)
                finalState = np.loadtxt(os.path.join(oDir, 'state.txt'))
                reward = np.loadtxt(os.path.join(oDir, 'reward.txt'))
                loss = np.loadtxt(os.path.join(oDir, 'loss.txt'))
                
                Npa = int( np.loadtxt(os.path.join(oDir, 'nPulse.txt')) ) # Number of pulse applications
                Nts = int( np.loadtxt(os.path.join(oDir, 'nTimeStep.txt')) )# Number of pulse chances/locations
                tMax = np.loadtxt(os.path.join(oDir, 'maxTime.txt'))
                chemPotential = np.loadtxt(os.path.join(oDir, 'noiseParam1.txt'))
                temperature = np.loadtxt(os.path.join(oDir, 'noiseParam2.txt'))


                # Get min and max temperatures
                if temperature < temp_min:
                    temp_min = temperature
                    S_temp_min = sOmega
                if temperature > temp_max:
                    temp_max = temperature
                    S_temp_max = sOmega

                pulse_timings, filters, overlaps, rewards = crunch_job(tMax,
                                                                      freq,
                                                                      sOmega,
                                                                      finalState,
                                                                      reward,
                                                                      weights=weights)
                plot_job(tMax,
                         freq,
                         sOmega,
                         pulse_timings,
                         Npa,
                         Nts,
                         filters,
                         overlaps,
                         rewards,
                         loss,
                         save = os.path.join(plot_dir, file),
                         show = False,
                         title = '$\mu = {0}, T = {1}$'.format(chemPotential, temperature))


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
    fig = plt.figure(layout = 'constrained', figsize = (16, 8))
    mosaic = """
             DA
             CB
             """
    axd = fig.subplot_mosaic(mosaic)
    
    UDDColor = '#377EB8'# Curious Blue
    UDDLinestyle = 'solid'
    UDDLabel = 'UDD'
    UDDMarker = 'x'

    agentColor = '#984EA3'# Deep Lilac
    agentLinestyle = 'solid'
    agentLabel = 'Agent' 
    
    CPMGColor = '#FF7F00'# Dark Orange
    CPMGLinestyle = 'solid'
    CPMGLabel = 'CPMG'
    CPMGMarker = '+'

    noiseColor = '#4DAF4A'# Fruit Salad, green
    noiseLinestyle = 'dashdot'
    noiseLabel = r'$|S(\nu)|^2$'
    
    
    # Plot overlaps. Use scatter since temperatures might be out of order due to file reading.
    axd['A'].scatter(temperatures, max_chi, color = agentColor, label = agentLabel)
    axd['A'].scatter(temperatures, UDD_chi, color = UDDColor, label = UDDLabel, marker = UDDMarker)
    axd['A'].scatter(temperatures, CPMG_chi, color = CPMGColor, label = CPMGLabel, marker = CPMGMarker)
    axd['A'].set_xlabel('Temperature')
    axd['A'].set_ylabel('$\chi$')
    axd['A'].legend()
    axd['A'].set_yscale('log')


    # Plot overlaps
    axd['B'].scatter(temperatures, max_rewards, color = agentColor, label = agentLabel)
    axd['B'].scatter(temperatures, UDD_rewards, color = UDDColor, label = UDDLabel, marker = UDDMarker)
    axd['B'].scatter(temperatures, CPMG_rewards, color = CPMGColor, label = CPMGLabel, marker = CPMGMarker)
    axd['B'].set_xlabel('Temperature')
    axd['B'].set_ylabel('$R(\chi)$')
    axd['B'].legend()
    axd['B'].set_yscale('log')

    # Plot noise across frequency with min and max temp
    axd['C'].set_title('$\mu =$ {:.3e}'.format(chemPotential))
    axd['C'].set_ylabel(r'$|S(\nu)|^2$')
    axd['C'].set_xlabel(r'$\nu$ in Hz')
    axd['C'].plot(freq, S_temp_min, label = '$T_{{min}}$ = {:.3e}'.format(temp_min), color = noiseColor)
    axd['C'].plot(freq, S_temp_max, label = '$T_{{max}}$ = {:.3e}'.format(temp_max), color = noiseColor)
    axd['C'].legend()

    # Plot the UDD and CPMG filter functions
    UDDFilter = filters[1]
    CPMGFilter = filters[2]
    axd['D'].set_ylabel(r'$F(\nu)$')
    axd['D'].set_xlabel(r'$\nu$ in Hz')
    axd['D'].plot(freq, UDDFilter, color = UDDColor, label = UDDLabel)
    axd['D'].plot(freq, CPMGFilter, color = CPMGColor, label = CPMGLabel)
    axd['D'].legend()
    
    plt.savefig(os.path.join(plot_dir, 'temperature.png'))
    if show:
        plt.show()

    print('done')
    return



if __name__=='__main__':
    
    # Load data

    oDir = '/home/charlie/Documents/ml/CollectiveAction/data/job_00001'


    nPulse = int( np.loadtxt(os.path.join(oDir, 'nPulse.txt')) ) # Number of pulse applications
    nTimeStep = int( np.loadtxt(os.path.join(oDir, 'nTimeStep.txt')) )# Number of pulse chances/locations
    tMax = np.loadtxt(os.path.join(oDir, 'maxTime.txt'))
    chemPotential = np.loadtxt(os.path.join(oDir, 'noiseParam1.txt'))
    temperature = np.loadtxt(os.path.join(oDir, 'noiseParam2.txt'))

    freq = np.loadtxt(os.path.join(oDir, 'freq.txt'))
    sOmega = np.loadtxt(os.path.join(oDir, 'sOmega.txt'))
    
    finalState = np.loadtxt(os.path.join(oDir, 'state.txt'))
    reward = np.loadtxt(os.path.join(oDir, 'reward.txt'))
    loss = np.loadtxt(os.path.join(oDir, 'loss.txt'))

    print('finalState')
    print(finalState)

    save = False # 'tplot.png'
    show = True
    #one_temp_one_agent(tMax,
    #                   chemPotential,
    #                   temperature,
    #                   freq,
    #                   sOmega,
    #                   finalState,
    #                   agentFilter,
    #                   reward,
    #                   loss,
    #                   save=save,
    #                   show=show)
   
    pulse_timings, filters, overlaps, rewards = crunch_job(tMax,
                                                          freq,
                                                          sOmega,
                                                          finalState,
                                                          reward)
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
             save = None, show = True, title = '$\mu = {0}, T = {1}$'.format(chemPotential, temperature))

    #dd = '/home/charlie/Documents/ml/Filter/temp_scan_data'
    #pd = '/home/charlie/Documents/ml/Filter/temp_scan_plots'
    #temperature_sweep(dd, pd, show=False)
