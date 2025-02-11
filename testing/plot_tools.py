import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pulse_sequence as ps

def kappa(t, harmonic, eta, max_time):
    return t + eta * np.sin(np.pi * harmonic * t / max_time)

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
    filters[0] = ps.FilterFunc(freq, pulse_timings[0], max_time) # Agent Filter 
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

    print("Agent reward python")
    print(ps.RewardFunc(overlaps[0], initial_chi))
    print("UDD reward")
    print(ps.RewardFunc(overlaps[1], initial_chi))
    print("CPMG reward")
    print(ps.RewardFunc(overlaps[2], initial_chi))


    print("Agent solutions")
    print(agent_state)
    print("UDD timings")
    print(pulse_timings[1])
    print("CPMG timings")
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
    
    UDDColor = '#377EB8'# Blue
    UDDLinestyle = 'solid'
    UDDLabel = r'$\chi_{UDD} = $' + '{:.3e}'.format(overlaps[1])

    agentColor = '#984EA3'# Lilac
    agentLinestyle = 'solid'
    agentLabel = r'$\chi_{Agent} = $' + '{:.3e}'.format(overlaps[0])
    
    CPMGColor = '#FF7F00'# Orange
    CPMGLinestyle = 'solid'
    CPMGLabel = r'$\chi_{CPMG} = $' + '{:.3e}'.format(overlaps[2])

    noiseColor = '#2CA02C'# Green
    noiseLinestyle = 'dashdot'
    noiseLabel = r'$S(\nu)$'

    lwps = 3 # Linewidth for pulse sequence plots
    # Find max index for plotting
    cutoffIdx = len(noise)-1 # Initialize
    for i in range(len(noise)-1, -1, -1):
        if (noise[i] / freq[i]**2 > 1e-3):
            cutoffIdx = i
            break
    print("cutoffFreq = {}".format(freq[cutoffIdx]))


    if title is not None:
        fig.suptitle(title)
    
    # Plot pulse sequence over time

    axd['A'].set_title('UDD ($N_{{pulse}} = {}$)'.format(int(Npa)))
    axd['A'].sharex(axd['C'])
    axd['A'].vlines(pulse_timings[1], 0, 1, color=UDDColor, linestyle=UDDLinestyle, linewidth = lwps)
    axd['A'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['A'].tick_params(labelbottom=False) # Remove x axis labels
    axd['A'].grid(axis = 'x')

    axd['B'].set_title('Agent ($N_{{time step}} = {0}, N_{{pulse}} = {1}$)'.format(int(Nts), int(Npa)))
    axd['B'].sharex(axd['C'])
    axd['B'].vlines(pulse_timings[0], 0, 1, color=agentColor, linewidth = lwps)
    axd['B'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['B'].tick_params(labelbottom=False) # Remove x axis labels
    axd['B'].grid(axis = 'x')
   
    axd['C'].set_title('CPMG ($N_{{pulse}} = {}$)'.format(int(Npa)))
    axd['C'].set_xlabel('Time')
    axd['C'].set_xlim(0, max_time)
    axd['C'].vlines(pulse_timings[2], 0, 1, color=CPMGColor, linestyle=CPMGLinestyle, linewidth=lwps)
    axd['C'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
    axd['C'].grid(axis = 'x')

    # Plot filter functions and noise PSD

    axd['D'].set_ylabel(r'$F(\nu)$')
    axd['D'].sharey(axd['F'])
    axUDDNoise = axd['D'].twinx()
    axUDDNoise.set_ylabel(noiseLabel)
    filterPlotUDD = axd['D'].plot(freq[:cutoffIdx], filters[1][:cutoffIdx], color=UDDColor, linestyle=UDDLinestyle, label = r'$F(\nu)_{UDD}$')
    noisePlotUDD = axUDDNoise.plot(freq[:cutoffIdx], noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\nu)$')
    linesUDD = filterPlotUDD + noisePlotUDD
    labelsUDD = [l.get_label() for l in linesUDD]
    axd['D'].tick_params(labelbottom=False) # Remove x axis labels
    axd['D'].set_title(UDDLabel)
    axd['D'].legend(linesUDD, labelsUDD)


    axd['E'].set_ylabel(r'$F(\nu)$')
    axd['E'].sharey(axd['F'])
    axAgentNoise = axd['E'].twinx()
    axAgentNoise.set_ylabel(noiseLabel)
    filterPlotAgent = axd['E'].plot(freq[:cutoffIdx], filters[0][:cutoffIdx], color=agentColor, linestyle=agentLinestyle, label = r'$F(\nu)_{Agent}$')
    noisePlotAgent = axAgentNoise.plot(freq[:cutoffIdx], noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\nu)$')
    linesAgent = filterPlotAgent + noisePlotAgent
    labelsAgent = [l.get_label() for l in linesAgent]
    axd['E'].tick_params(labelbottom=False) # Remove x axis labels
    axd['E'].set_title(agentLabel)
    axd['E'].legend(linesAgent, labelsAgent)

    axd['F'].set_ylabel(r'$F(\nu)$')
    axCPMGNoise = axd['F'].twinx()
    axCPMGNoise.set_ylabel(noiseLabel)
    filterPlotCPMG = axd['F'].plot(freq[:cutoffIdx], filters[2][:cutoffIdx], color=CPMGColor, linestyle=CPMGLinestyle, label = r'$F(\nu)_{CPMG}$')
    noisePlotCPMG = axCPMGNoise.plot(freq[:cutoffIdx], noise[:cutoffIdx], color=noiseColor, linestyle=noiseLinestyle, label = r'$S(\nu)$')
    linesCPMG = filterPlotCPMG + noisePlotCPMG
    labelsCPMG = [l.get_label() for l in linesCPMG]
    axd['F'].set_title(CPMGLabel)
    axd['F'].set_xlabel(r'$\nu$ [1/time]')
    axd['F'].legend(linesCPMG, labelsCPMG)
    axd['F'].set_yscale(filterScale)

    axUDDNoise.sharey(axCPMGNoise)
    axAgentNoise.sharey(axCPMGNoise)
    axCPMGNoise.set_yscale(noiseScale)

    # Plot reward and loss over trials
    
    nTrial = rewards.shape[1]
    trials = np.arange(1, nTrial + 1) # Do arange(1, n+1) so that index starts at 1, ends at nTrial
    axd['G'].set_ylabel('Reward')
    axd['G'].sharex(axd['H'])
    axd['G'].plot(trials, rewards[0], color=agentColor, label = 'Agent') 
    axd['G'].plot(trials, rewards[1], color=UDDColor, linestyle=UDDLinestyle, label = 'UDD') 
    axd['G'].plot(trials, rewards[2], color=CPMGColor, linestyle=CPMGLinestyle, label = 'CPMG')
    #axd['G'].set_yscale('log')

    # Get index of maximum rewards
    idxMax = np.argmax(rewards[0])
    axd['G'].scatter(trials[idxMax], rewards[0][idxMax], color=agentColor, label = 'Agent Max Reward', facecolors='none')
    axd['G'].legend()

    axd['H'].set_ylabel('Average Loss')
    axd['H'].set_xlabel('Trials')
    axd['H'].plot(trials, agent_loss, color=agentColor)

    #fig.tight_layout()
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
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

    # Check if plot directory exists
    if os.path.exists(plot_dir) == False:
        raise ValueError("Path " + plot_dir + " does not exist.")

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
                initialState = np.loadtxt(os.path.join(oDir, 'initialState.txt'))
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
                                                                      initialState,
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
    
    UDDColor = '#1F77B4'#'#377EB8'# Curious Blue
    UDDLinestyle = 'solid'
    UDDLabel = 'UDD'
    UDDMarker = 'x'

    agentColor = '#984EA3'# Deep Lilac
    agentLinestyle = 'solid'
    agentLabel = 'Agent' 
    
    CPMGColor = '#FF7F0E'#'#FF7F00'# Dark Orange
    CPMGLinestyle = 'solid'
    CPMGLabel = 'CPMG'
    CPMGMarker = '+'

    noiseColor = '#2CA02C'#'#4DAF4A'# Fruit Salad, green
    noiseLinestyle = 'dashdot'
    noiseLabel = r'$|S(\nu)|^2$'

    cutoffIdx = len(S_temp_max) - 1 # Initialize
    # Find max index for plotting
    for i in range(cutoffIdx, -1, -1):
        if (S_temp_max[i] / freq[i]**2 > 1e-3):
            cutoffIdx = i
            break
    
    print("cutoffIdx: {}".format(cutoffIdx))

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
    axd['C'].plot(freq[:cutoffIdx], S_temp_min[:cutoffIdx], label = '$T_{{min}}$ = {:.3e}'.format(temp_min), color = noiseColor, linestyle = 'dashed')
    axd['C'].plot(freq[:cutoffIdx], S_temp_max[:cutoffIdx], label = '$T_{{max}}$ = {:.3e}'.format(temp_max), color = noiseColor)
    axd['C'].legend()

    # Plot the UDD and CPMG filter functions
    UDDFilter = filters[1]
    CPMGFilter = filters[2]
    axd['D'].set_ylabel(r'$F(\nu)$')
    axd['D'].set_xlabel(r'$\nu$ in Hz')
    axd['D'].plot(freq[:cutoffIdx], UDDFilter[:cutoffIdx], color = UDDColor, label = UDDLabel)
    axd['D'].plot(freq[:cutoffIdx], CPMGFilter[:cutoffIdx], color = CPMGColor, label = CPMGLabel)
    axd['D'].legend()
    
    plt.savefig(os.path.join(plot_dir, 'temperature.png'))
    if show:
        plt.show()

    print('done')
    return


def single_eta_multi_run(data_dir, subdir_prefix='run', show = True, save = None):
    """
    PARAMETERS:
    data_dir (string): directory containing the subdirectories which contain the data files
    subdir_prefix (Optional, string): the prefix of all the subdirectories
    show (bool): indicates whether to show the eta plot using matplotlib's interactive environment

    RETURNS:
    None
    """

    ## Check if plot directory exists
    #if os.path.exists(plot_dir) == False:
    #    raise ValueError("Path " + plot_dir + " does not exist.")

    # Lists will store rewards over eta
    infidelity_list = []
    final_state_list = []
    loss_list = []

    run_counter = 0
    file_fail_counter = 0

    print("Reading from directory "+ data_dir)
    Nfiles = len(os.listdir(data_dir))
    for file in os.listdir(data_dir):
        if file.startswith(subdir_prefix):

            try:
                oDir = os.path.join(data_dir, file)

                # Load data

                freq = np.loadtxt(os.path.join(oDir, 'freq.txt'))
                sOmega = np.loadtxt(os.path.join(oDir, 'sOmega.txt'))

                finalState = np.loadtxt(os.path.join(oDir, 'state.txt'))
                rewards = np.loadtxt(os.path.join(oDir, 'reward.txt'))
                loss = np.loadtxt(os.path.join(oDir, 'loss.txt'))
                
                nPulse = int( np.loadtxt(os.path.join(oDir, 'nPulse.txt')) ) # Number of pulse applications
                tMax = np.loadtxt(os.path.join(oDir, 'maxTime.txt'))
                etaN = np.loadtxt(os.path.join(oDir, 'etaN.txt'))
                harmonics = np.loadtxt(os.path.join(oDir, 'harmonics.txt')).astype(int)

                infidelities = 1 - ps.fid_from_reward(rewards)

                infidelity_list.append(infidelities)
                final_state_list.append(finalState)
                loss_list.append(loss)
                
                print("Run {} loaded".format(run_counter))

                run_counter += 1
            
            except FileNotFoundError:
                raise FileNotFoundError(f"{file} not found or is missing data.")
        else:
            print(f"File {file} is not a target file. Trying next file.")
            file_fail_counter += 1

    if file_fail_counter == Nfiles:
        raise Exception("No target files found.")
                

    UDD_times = ps.UDD(nPulse, tMax) 
    CPMG_times = ps.CPMG(nPulse, tMax) 
    UDD_filter = ps.FilterFunc(freq, UDD_times, tMax) # Agent Filter 
    CPMG_filter = ps.FilterFunc(freq, CPMG_times, tMax) # UDD filter 
    UDD_chi = ps.chi(freq, sOmega, UDD_filter)
    CPMG_chi = ps.chi(freq, sOmega, CPMG_filter)
    UDD_infid = 1 - ps.fidelity(UDD_chi)
    CPMG_infid = 1 - ps.fidelity(CPMG_chi)

    print("UDD infidelity: ", UDD_infid)
    print("CPMG infidelity: ", CPMG_infid)

    # Plot chi and reward over eta
    fig = plt.figure(layout = 'constrained', figsize = (16, 8))
    mosaic = """
             AB
             AC
             """
    axd = fig.subplot_mosaic(mosaic)
    
    UDDColor = '#1F77B4'#'#377EB8'# Curious Blue
    UDDLinestyle = 'dashed'
    UDDLabel = 'UDD'

    CPMGColor = '#FF7F0E'#'#FF7F00'# Dark Orange
    CPMGLinestyle = 'dashed'
    CPMGLabel = 'CPMG'

    pulse_seq_lw = 2 # Linewidth for pulse sequences in the plot.

    # X axis for axis X
    axd['A'].set_xlim( (0, tMax) )
    xtick_positions = [0, tMax/4, tMax/2, 3*tMax/4, tMax]
    xtick_labels = ['0', '0.25T', '0.5T', '0.75T', 'T']
    axd['A'].set_xticks(ticks=xtick_positions, labels=xtick_labels)
    axd['A'].set_xlabel('Time')

    # Parameters for vlines
    nRun = len(infidelity_list)
    vlines_ymin = np.arange(nRun)
    vlines_ymax = vlines_ymin + 1

    # Lefthand Y axis
    axd['A'].set_ylim((-1, nRun + 1))
    y1tick_locs = vlines_ymin + 0.5
    y1tick_labels = np.arange(1, nRun+1)
    axd['A'].set_yticks(ticks=y1tick_locs, labels = y1tick_labels)
    axd['A'].set_ylabel('Runs')

    # Make infidelity and loss plots
    nTrial = len(infidelities)
    trials = np.arange(1, nTrial + 1) # Do arange(1, n+1) so that index starts at 1, ends at nTrial
    axd['B'].set_ylabel('Infidelity')
    axd['C'].set_ylabel('Average Loss')
    axd['B'].sharex(axd['C'])
    axd['C'].set_xlabel('Trials')
    for i in range(nRun):
        axd['A'].vlines(final_state_list[i], vlines_ymin[i], vlines_ymax[i], lw=pulse_seq_lw)
        axd['B'].plot(trials, infidelity_list[i], label = 'Run {}'.format(i)) 
        axd['C'].plot(trials, loss_list[i], label = 'Run {}'.format(i)) 
    axd['B'].plot(trials, np.full(nTrial, UDD_infid), color=UDDColor, linestyle=UDDLinestyle, label = 'UDD') 
    axd['B'].plot(trials, np.full(nTrial, CPMG_infid), color=CPMGColor, linestyle=CPMGLinestyle, label = 'CPMG')
    #axd['B'].set_yscale('log')

    # Get index of maximum rewards
    axd['B'].legend()

    # Set title
    plt.suptitle('$\eta_{{{0}}}$ = {1}, Harmonic Set: {2}'.format(nPulse, etaN, harmonics))

    #fig.tight_layout()
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
    plt.close()

    return


def multi_eta_multi_run(data_dir, subdir_prefix='job', subsubdir_prefix='run', show = True, save = None):
    """
    Make plot of eta on the x axis and fidelity on the y axis.
    PARAMETERS:
    data_dir (string): directory containing the subdirectories which contain the data files
    subdir_prefix (Optional, string): the prefix of all the subdirectories
    show (bool): indicates whether to show the eta plot using matplotlib's interactive environment

    RETURNS:
    None
    """

    ## Check if plot directory exists
    #if os.path.exists(plot_dir) == False:
    #    raise ValueError("Path " + plot_dir + " does not exist.")

    # Lists will store values corresponding to each eta
    eta_list = []
    avg_min_infid_list = [] # List of average minimum infidelities found in each job directory
    std_min_infid_list = [] # List of standard deviation of infidelities found in each job directory


    print("Reading from directory "+ data_dir)
    job_counter = 0
    # For every file in data_dir
    for file in os.listdir(data_dir):
        if file.startswith(subdir_prefix):
            file_fail_counter = 0
            Nfiles = len(os.listdir(os.path.join(data_dir, file))) # Number files in subdirectory
            min_infid_list = [] # List of average minimum infidelities found in each job directory
            etaN_list = []

            run_counter = 0
            # For every file in the subdirectory
            for subdir_file in os.listdir(os.path.join(data_dir, file)):
                if subdir_file.startswith(subsubdir_prefix):
                    try:
                        oDir = os.path.join(data_dir, file, subdir_file) # Directory with output data

                        # Load data
                        if run_counter == 0:
                            etaN = np.loadtxt(os.path.join(oDir, 'etaN.txt')).item()
                            eta_list.append(etaN)
                        if job_counter == 0:
                            harmonics = np.loadtxt(os.path.join(oDir, 'harmonics.txt')).astype(int)
                            freq = np.loadtxt(os.path.join(oDir, 'freq.txt'))
                            sOmega = np.loadtxt(os.path.join(oDir, 'sOmega.txt'))
                            nPulse = int( np.loadtxt(os.path.join(oDir, 'nPulse.txt')) ) # Number of pulse applications
                            tMax = np.loadtxt(os.path.join(oDir, 'maxTime.txt'))
                            
                        rewards = np.loadtxt(os.path.join(oDir, 'reward.txt'))
                        infidelities = 1 - ps.fid_from_reward(rewards)
                        min_infid_list.append(np.min(infidelities))
                        
                        run_counter += 1
                    
                    except FileNotFoundError:
                        raise FileNotFoundError(f"{oDir} not found or is missing data.")
                else:
                    print(f"File {oDir} is not a target file. Trying next file.")
                    file_fail_counter += 1

            if file_fail_counter == Nfiles:
                raise Exception("No target files found in {os.path.join(data_dir, file)}.")
        avg_min_infid_list.append( np.mean(np.array(min_infid_list)) )
        std_min_infid_list.append( np.std(np.array(min_infid_list)) )
        job_counter += 1

    UDD_times = ps.UDD(nPulse, tMax) 
    CPMG_times = ps.CPMG(nPulse, tMax) 
    UDD_filter = ps.FilterFunc(freq, UDD_times, tMax) # Agent Filter 
    CPMG_filter = ps.FilterFunc(freq, CPMG_times, tMax) # UDD filter 
    UDD_chi = ps.chi(freq, sOmega, UDD_filter)
    CPMG_chi = ps.chi(freq, sOmega, CPMG_filter)
    UDD_infid = 1 - ps.fidelity(UDD_chi)
    CPMG_infid = 1 - ps.fidelity(CPMG_chi)

    print("UDD infidelity: ", UDD_infid)
    print("CPMG infidelity: ", CPMG_infid)

    ## Plot chi and reward over eta
    #fig = plt.figure(layout = 'constrained', figsize = (16, 8))
    #mosaic = """
    #         AB
    #         AC
    #         """
    #axd = fig.subplot_mosaic(mosaic)
    
    UDDColor = '#1F77B4'#'#377EB8'# Curious Blue
    UDDLinestyle = 'dashed'
    UDDLabel = 'UDD'

    CPMGColor = '#FF7F0E'#'#FF7F00'# Dark Orange
    CPMGLinestyle = 'dashed'
    CPMGLabel = 'CPMG'
    
    agentColor = '#984EA3'# Deep Lilac
    agentLinestyle = 'solid'
    agentLabel = 'Agent' 

    pulse_seq_lw = 2 # Linewidth for pulse sequences in the plot.
    
    Neta = len(eta_list)
    # Sort values because sometimes data gets loaded out of order
    sort_indices = np.argsort(eta_list)
    eta_arr = np.array(eta_list)[sort_indices]
    avg_min_infid_arr = np.array(avg_min_infid_list)[sort_indices]
    std_min_infid_arr = np.array(std_min_infid_list)[sort_indices]

    # Plot
    plt.errorbar(eta_arr, avg_min_infid_arr, yerr=std_min_infid_arr, color = agentColor, label='Harmonics: {}'.format(harmonics))
    plt.plot(eta_list, np.full(Neta, CPMG_infid), color = CPMGColor, label='CPMG')
    plt.plot(eta_list, np.full(Neta, UDD_infid), color = UDDColor, label='UDD')
    plt.ylabel('Average of Minimum Infidelity')
    plt.xlabel('$\eta_{{{}}}$'.format(nPulse))
    plt.title('Max Step = {}'.format(20))

    plt.legend()

    plt.show()


    return

def multi_single_eta_plots(data_dir, save, subdir_prefix='job'):
    """
    Make many single eta plots.

    PARAMETERS:
    data_dir (string): directory containing the subdirectories which contain the data files
    subdir_prefix (Optional, string): the prefix of all the subdirectories
    show (bool): indicates whether to show the eta plot using matplotlib's interactive environment

    RETURNS:
    None
    """

    ## Check if plot directory exists
    #if os.path.exists(plot_dir) == False:
    #    raise ValueError("Path " + plot_dir + " does not exist.")

    run_counter = 0
    file_fail_counter = 0

    print("Reading from directory "+ data_dir)
    Nfiles = len(os.listdir(data_dir))
    for file in os.listdir(data_dir):
        if file.startswith(subdir_prefix):

            try:
                oDir = os.path.join(data_dir, file)
                single_eta_multi_run(oDir, show=False, save=os.path.join(save, file))
                
                print("Run {} loaded".format(run_counter))

                run_counter += 1
            
            except FileNotFoundError:
                raise FileNotFoundError(f"{file} not found or is missing data.")
        else:
            print(f"File {file} is not a target file. Trying next file.")
            file_fail_counter += 1

    if file_fail_counter == Nfiles:
        raise Exception("No target files found.")

    return

if __name__=='__main__':
     
    # Load data
    """
    oDir = '/home/charlie/Documents/ml/CollectiveAction/data/job_00000'

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

    save = False # 'tplot.png'
    show = True
   
    pulse_timings, filters, overlaps, rewards = crunch_job(tMax,
                                                          freq,
                                                          sOmega,
                                                          finalState,
                                                          reward,
                                                          initialState)

    #job_title = '$\mu = {0}, T = {1}$'.format(noiseParam1, noiseParam2)
    job_title = '$bandCenter = {0}*cpmgPeak, bandWidth = {1} / maxTime$'.format(noiseParam2, noiseParam1)
    #job_title = 'Sum of Lorentzians on CPMG Peaks'

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
             save = None, show = True, title = job_title,
             filterScale = 'linear', noiseScale = 'linear')
    """

    
    dd = '/home/charlie/Documents/ml/CollectiveAction/eta_scan_data/1_over_f/harmonics_01_02_08/job_00001'
    pd = '/home/charlie/Documents/ml/CollectiveAction/paper_plots'
    #temperature_sweep(dd, pd, show=False)
    single_eta_multi_run(dd, show=True)
    #multi_eta_multi_run(dd, show=True)
    #multi_single_eta_plots(dd, save=pd)
