import numpy as np
import os
import plot_tools as pt
import pulse_sequence as ps

# Load data
#oDir = '/home/charlie/Documents/ml/CollectiveAction/data'
oDir = '/home/charlie/Documents/ml/CollectiveAction/data_NJ_scan_lorentzians_run1/N_064/J_007'
#oDir = '/home/charlie/Documents/ml/CollectiveAction/bootstrap_testing/Npulse_8/seed_N_8_j_1_2_7_8/data_N_8_j_1_2_9_10'

nPulse = int( np.loadtxt(os.path.join(oDir, 'nPulse.txt')) ) # Number of pulse applications
nTimeStep = int( np.loadtxt(os.path.join(oDir, 'nTimeStep.txt')) )# Number of pulse chances/locations
tMax = np.loadtxt(os.path.join(oDir, 'maxTime.txt'))
noiseParam1 = np.loadtxt(os.path.join(oDir, 'noiseParam1.txt'))
noiseParam2 = np.loadtxt(os.path.join(oDir, 'noiseParam2.txt'))

freq = np.loadtxt(os.path.join(oDir, 'freq.txt'))
weights = np.loadtxt(os.path.join(oDir, 'weights.txt'))
sOmega = np.loadtxt(os.path.join(oDir, 'sOmega.txt'))

finalState = np.loadtxt(os.path.join(oDir, 'state.txt'))
reward = np.loadtxt(os.path.join(oDir, 'reward.txt'))
loss = np.loadtxt(os.path.join(oDir, 'loss.txt'))
initialState = np.loadtxt(os.path.join(oDir, 'initialState.txt'))

print('finalState')
print(finalState)

#save = None#'/home/charlie/Documents/ml/CollectiveAction/plots/job_00000.png'
save = '/home/charlie/Documents/ml/CollectiveAction/bootstrap_testing/Npulse_8/seed_N_8_j_1_2_7_8/data_N_8_j_1_2_9_10.svg'
show = True

pulse_timings, filters, overlaps, rewards = pt.crunch_job(tMax,
                                                      freq,
                                                      sOmega,
                                                      finalState,
                                                      reward,
                                                      initialState,
                                                      weights=weights)

#job_title = '$\mu = {0}, T = {1}$'.format(noiseParam1, noiseParam2)
#job_title = '$bandCenter = {0}*cpmgPeak, bandWidth = {1} / maxTime$'.format(noiseParam2, noiseParam1)
#job_title = 'Sum of Lorentzians'
job_title = '1/f noise with Low Frequency Cutoff'

fidelities = ps.fidelity(overlaps)
print('Agent Fidelity: ', fidelities[0])
print('UDD Fidelity: ', fidelities[1])
print('CPMG Fidelity: ', fidelities[2])

pt.plot_job(tMax,
            freq,
            sOmega,
            pulse_timings,
            nPulse,
            nTimeStep,
            filters,
            overlaps,
            rewards,
            loss,
            save = save, show = show, title = job_title,
            filterScale = 'linear', noiseScale = 'linear')
