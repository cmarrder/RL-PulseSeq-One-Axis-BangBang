import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

max_time = 1
harmonic_list = [0, 1, -1, 2, -2, 3, -3]

def nonlinear_action(x, harmonic, highest_harmonic, eta):
    """
    PARAMETERS:
    x (np.ndarray of shape (1,)): abscissa points to calculate
    harmonic (int): the harmonic of the current action
    highest_harmonic (int): positive integer. abs(harmonic) should be less than this.
    eta (float): nonlinearity parameter. Should be less than abs(harmonic) / highest_harmonic.
    RETURNS:
    y (np.ndarray of shape (1,)): ordinate points of function
    """
    if eta >  highest_harmonic / abs(harmonic) or eta < 0:
        raise ValueError('eta is outside its domain of validity, [0, highest_harmonic / abs(harmonic) ].')
    action_interval = np.pi * highest_harmonic
    shifted_x = x * (action_interval / max_time)
    y = shifted_x + eta * np.sin((harmonic / highest_harmonic) * shifted_x)
    y = y * (max_time / action_interval)
    return y

def linear_flip_action(x):
    """
    PARAMETERS:
    x (np.ndarray of shape (1,)): (ordered) abscissa points to calculate
    RETURNS:
    y (np.ndarray of shape (1,)): ordinate points of function
    """
    y = max_time - x
    return y

def kappa(t, j, eta):
    """
    PARAMETERS:
    t (np.ndarray): abscissa points to apply function to
    j (int): harmonic label
    eta (float): nonlinearity parameter. Should be greater than 0, less than max_time / pi / abs(j).
    """
    if eta >= max_time / np.pi / abs(j) or eta <= 0:
        raise ValueError('eta is outside its domain of validity, (0, max_time / pi / abs(j) ).')
    return t + eta * sin(np.pi * j * (t / max_time))

def make_action_plot(x, title, color, *args):
    """
    Illustrate action functions for the PAPER.
    PARAMETERS:
    x (np.ndarray of shape (1,)): abscissa points to calculate for action function. Assuming starting from 0 and going to maxtime.
    func (function): action function to plot
    args (listlike): arguments for func
    RETURNS:
    fig (plt.figure)
    """

    y = func(x, *args)

    # Size for lines in plot for the PAPER.
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['xtick.major.size'] = 7#5
    mpl.rcParams['xtick.major.width'] = 2 
    mpl.rcParams['xtick.minor.size'] = 1
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.major.size'] = 7#5
    mpl.rcParams['ytick.major.width'] = 2 
    mpl.rcParams['ytick.minor.size'] = 1
    mpl.rcParams['ytick.minor.width'] = 1
    plt.rcParams['figure.constrained_layout.use'] = True

    fig, ax = plt.subplots()

    tick_positions = [x[0], x[-1]/2, x[-1]] # Ticks at 0, 0.5*maxtime, maxtime
    tick_labels = ['0', '0.5T', 'T']
    plt.xticks(ticks=tick_positions, labels=tick_labels)
    plt.yticks(ticks=tick_positions, labels=tick_labels)

    # Define font sizes
    SIZE_DEFAULT = 16
    SIZE_LARGE = 18

    # Define pad sizes
    PAD_DEFAULT = 10

    ax.set_xlabel('Old Times', fontsize = SIZE_LARGE, labelpad=PAD_DEFAULT)
    ax.set_ylabel('New Times', fontsize = SIZE_LARGE, labelpad=PAD_DEFAULT)
    ax.tick_params(axis='both', which='major', labelsize=SIZE_DEFAULT)
    ax.set_aspect('equal')

    # Define linewidth size
    lw = 3

    if xsample is not None:
        ysample = func(xsample)
        ax.vlines(xsample, np.zeros_like(xsample), ysample, linewidth=lw/2, linestyles='dashed')
        ax.hlines(ysample, np.zeros_like(ysample), xsample, linewidth=lw/2, linestyles='dashed')

    if title is not None:
        ax.set_title(title, fontsize = SIZE_LARGE, pad=PAD_DEFAULT)

    if color is not None:
        ax.plot(x, y, color = color, linewidth = lw)
    else:
        ax.plot(x, y, linewidth = lw)
    
    return fig


def make_action_list():
    """
    Make a list of lambda functions, each one a wrapper for one of the action functions.
    PARAMETERS:
    None
    RETURNS:
    list of function objects
    """
    # HERE
    act_list = []
    highestHarmonic = 4
    harmonics = [1, -1, 2, -2, 4, -4]
    eta_array = highestHarmonic / abs(np.array(harmonics)) # Make each eta value the max possible eta value
    # For some reason trying to use a list and the append method doesn't work. It seems
    # to only use the final lambda function added in the loop. Hard coding for now.
    func1 = lambda x: nonlinear_action(x, 1, highestHarmonic, eta_array[0])
    func2 = lambda x: nonlinear_action(x, -1, highestHarmonic, eta_array[1])
    func3 = lambda x: nonlinear_action(x, 2, highestHarmonic, eta_array[2])
    func4 = lambda x: nonlinear_action(x, -2, highestHarmonic, eta_array[3])
    func5 = lambda x: nonlinear_action(x, 4, highestHarmonic, eta_array[4])
    func6 = lambda x: nonlinear_action(x, -4, highestHarmonic, eta_array[5])
    func7 = lambda x: linear_flip_action(x)
    act_list = [func1, func2, func3, func4, func5, func6, func7]
    return act_list

def plot_action_sequence(action_idxs):
    """
    PARAMETERS:
    action_idxs (list): list of integers corresponding to each action
    RETURNS:
    None
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
    
    # Make time mesh
    x = np.linspace(0, max_time, 1000)
    # Make initial sequence PDD of length 6
    Nsample = 3 
    xsample = np.linspace(x[0], x[-1], Nsample+2)[1:Nsample+1]

    # Make list of action functions
    action_list = make_action_list()

    actionColor = '#D62728' # Red
    agentColor = '#984EA3'# Lilac

    lw = 5 # Define linewidth size for action plot
    lwps = 5 # Linewidth for pulse sequence plots
    tick_positions = [x[0], x[-1]/2, x[-1]] # Ticks at 0, 0.5*maxtime, maxtime
    tick_labels = ['0', '0.5T', 'T']

    for i, a in enumerate(action_idxs):
        afunc = action_list[a]
    
        fig = plt.figure(layout = 'constrained', figsize = (18, 6.5))
        mosaic = """
                 .B.
                 ABC
                 .B.
                 """
        axd = fig.subplot_mosaic(
                mosaic,
                width_ratios=[1, 1, 1],
                #)
                gridspec_kw={
                    'wspace': 0.2, # Padding between columns
                    })
       
        fig.suptitle('Step {}'.format(i), fontsize=1.5*SIZE_LARGE)

        # Plot pulse sequence over time
        axd['A'].set_title('$s_{}$'.format(i), fontsize = SIZE_LARGE, pad=PAD_DEFAULT)
        axd['A'].vlines(xsample, 0, 1, color=actionColor, linewidth = lwps)
        axd['A'].set_xticks(ticks=tick_positions, labels=tick_labels)
        axd['A'].tick_params(axis='x', which='major', labelsize=SIZE_DEFAULT)
        axd['A'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks
        
        y = afunc(x) 
        ysample = afunc(xsample) # Dotted lines where sample is
        sample_linestyle = (0, (1,1))#'dotted'
        axd['B'].vlines(xsample, np.zeros_like(xsample), ysample, color=actionColor, linewidth=0.7*lw, linestyles=sample_linestyle)
        axd['B'].hlines(ysample, np.zeros_like(ysample), xsample, color=actionColor, linewidth=0.7*lw, linestyles=sample_linestyle)
        axd['B'].set_xticks(ticks=tick_positions, labels=tick_labels)
        axd['B'].set_yticks(ticks=tick_positions, labels=tick_labels)
        axd['B'].tick_params(axis='both', which='major', labelsize=SIZE_DEFAULT)
        # Plot action function curve
        axd['B'].set_title('$a_{}$'.format(i), fontsize = SIZE_LARGE, pad=PAD_DEFAULT)
        axd['B'].plot(x, y, color = agentColor, linewidth = lw)
        axd['B'].set_aspect('equal')
        axd['B'].set_xlabel('Old Times', fontsize = SIZE_LARGE, labelpad=PAD_DEFAULT)
        axd['B'].set_ylabel('New Times', fontsize = SIZE_LARGE, labelpad=PAD_DEFAULT)

        # Update xsample
        xsample = ysample

        # Plot pulse sequence over time
        axd['C'].set_title('$s_{}$'.format(i+1), fontsize = SIZE_LARGE, pad=PAD_DEFAULT)
        axd['C'].vlines(xsample, 0, 1, color=actionColor, linewidth = lwps)
        axd['C'].set_xticks(ticks=tick_positions, labels=tick_labels)
        axd['C'].tick_params(axis='x', which='major', labelsize=SIZE_DEFAULT)
        axd['C'].yaxis.set_major_locator(ticker.NullLocator()) # Remove y axis labels/ticks

        fname = 'step{}'.format(i) + '.png'
        path = '/home/charlie/Documents/ml/CollectiveAction/paper_plots'
        plt.savefig(os.path.join(path, fname), dpi = 300)
        plt.close(fig=fig)
        plt.show()
    
    return
    

if __name__=="__main__":
    """
    x = np.linspace(0, max_time, 1000)
    c = '#984EA3' # Deep Lilac
    action_list = make_action_list()

    # Make initial sequence PDD of length 6
    Nsample = 8
    xsample = np.linspace(x[0], x[-1], Nsample)[1:7]
     
    for k, action in enumerate(action_list):
        title = 'Action {}'.format(k+1)
        fig = make_action_plot_paper(x, action, xsample, title, c)
        #fname = 'action{}'.format(k+1) + '.png'
        #plt.savefig(fname, dpi = 300)
        #plt.close(fig=fig)
        plt.show()
    """ 
    plot_action_sequence([0,2,6])
