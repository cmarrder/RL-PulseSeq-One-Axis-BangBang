import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import harmonic_actions as ha
import pulse_sequence as ps

max_time = 1

def kappa(t, j, eta):
    """
    PARAMETERS:
    t (np.ndarray): abscissa points to apply function to
    j (int): harmonic label
    eta (float): nonlinearity parameter. Should be greater than 0, less than max_time / pi / abs(j).
    """
    if eta >= max_time / np.pi / abs(j) or eta <= 0:
        raise ValueError('eta is outside its domain of validity, (0, max_time / pi / abs(j) ).')
    return t + eta * np.sin(np.pi * j * (t / max_time))


def make_action_plot_poster(x, func, title, color):
    """
    Illustrate action functions for the POSTER.
    PARAMETERS:
    x (np.ndarray of shape (1,)): abscissa points to calculate for action function
    func (function): action function to plot
    args (listlike): arguments for func
    RETURNS:
    fig (plt.figure)
    """

    y = func(x)

    # Size for lines in plot for the POSTER.
    mpl.rcParams['axes.linewidth'] = 6#2
    mpl.rcParams['xtick.major.size'] = 15#5
    mpl.rcParams['xtick.major.width'] = 6#2 
    mpl.rcParams['xtick.minor.size'] = 3#1
    mpl.rcParams['xtick.minor.width'] = 3#1
    mpl.rcParams['ytick.major.size'] = 15#5
    mpl.rcParams['ytick.major.width'] = 6#2 
    mpl.rcParams['ytick.minor.size'] = 3#1
    mpl.rcParams['ytick.minor.width'] = 3#1
    plt.rcParams['figure.constrained_layout.use'] = True

    fig, ax = plt.subplots()

    ticks = np.linspace(x[0], x[-1], 3)
    plt.xticks(ticks=ticks)
    plt.yticks(ticks=ticks)
    #tick_positions = [x[0], x[-1]/2, x[-1]] # Ticks at 0, 0.5*maxtime, maxtime
    #tick_labels = ['0', '0.5T', 'T']
    #plt.xticks(ticks=tick_positions, labels=tick_labels)
    #plt.yticks(ticks=tick_positions, labels=tick_labels)

    # Define font sizes
    SIZE_DEFAULT = 24#12
    SIZE_LARGE = 32#16

    # Define pad sizes
    PAD_DEFAULT = 10

    ax.set_xlabel('Old Times', fontsize = SIZE_LARGE, labelpad=PAD_DEFAULT, weight='bold')
    ax.set_ylabel('New Times', fontsize = SIZE_LARGE, labelpad=PAD_DEFAULT, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=SIZE_DEFAULT)
    ax.set_aspect('equal')

    # Define linewidth size
    lw = 9#3

    Nsample = 8
    xsample = np.linspace(x[0], x[-1], Nsample)[1:7]
    ysample = func(xsample)
    ax.vlines(xsample, np.zeros_like(xsample), ysample, linewidth=lw/4, linestyles='dashed')
    ax.hlines(ysample, np.zeros_like(ysample), xsample, linewidth=lw/4, linestyles='dashed')

    if title is not None:
        ax.set_title(title, fontsize = SIZE_LARGE, pad=PAD_DEFAULT, weight='bold')

    if color is not None:
        ax.plot(x, y, color = color, linewidth = lw)
    else:
        ax.plot(x, y, linewidth = lw)
    
    return fig

def plot_1_kappa_paper(x, harmonic, eta, xsample, title, color, ax=None, save=None, show=True):
    """
    Illustrate action functions for the PAPER.
    PARAMETERS:
    x (np.ndarray of shape (1,)): abscissa points to calculate for action function. Assuming starting from 0 and going to maxtime.
    func (function): action function to plot
    args (listlike): arguments for func
    RETURNS:
    fig (plt.figure)
    """

    y = kappa(x, harmonic, eta)

    if ax is None:
        fig, ax = plt.subplots()

    #tick_positions = [x[0], x[-1]/2, x[-1]] # Ticks at 0, 0.5*maxtime, maxtime
    #tick_labels = ['0', '0.5T', 'T']
    tick_positions = [x[0], x[-1]] # Ticks at 0, 0.5*maxtime, maxtime
    tick_labels = ['0', 'T']
    ax.set_xticks(ticks=tick_positions, labels=tick_labels)
    ax.set_yticks(ticks=tick_positions, labels=tick_labels)

    # Define font sizes
    SIZE_DEFAULT = 24
    SIZE_LARGE = 36#24#18

    # Define pad sizes
    PAD_DEFAULT = 0#10
    PAD_LARGE = 15

    ax.set_xlabel('Old Times', fontsize = SIZE_DEFAULT, labelpad=PAD_DEFAULT)
    ax.set_ylabel('New Times', fontsize = SIZE_DEFAULT, labelpad=PAD_DEFAULT)
    ax.tick_params(axis='both', which='major', labelsize=SIZE_DEFAULT)
    ax.set_aspect('equal')

    # Define linewidth size
    lw = 6

    if xsample is not None:
        ysample = kappa(xsample, harmonic, eta)
        ax.vlines(xsample, np.zeros_like(xsample), ysample, linewidth=lw/2, linestyles='dashed')
        ax.hlines(ysample, np.zeros_like(ysample), xsample, linewidth=lw/2, linestyles='solid')

    if title is not None:
        ax.set_title(title, fontsize = SIZE_LARGE, pad=PAD_LARGE)

    if color is not None:
        ax.plot(x, y, color = color, linewidth = lw)
    else:
        ax.plot(x, y, linewidth = lw)

    if save is not None:
        plt.savefig(save)
    if show is True:
        plt.show()
    
    return

def plot_3_kappas_paper(x, xsample, color, save=None, show=True):
    fig = plt.figure(layout='constrained', figsize=(12, 4))
    mosaic = """
             ABC
             """
    axd = fig.subplot_mosaic(mosaic) 
    axd_keys = ['A', 'B', 'C']
    harmonics = [1, -2, 3]
    for key, j in zip(axd_keys, harmonics):
        eta = ha.eta_exclusive_bounds(j, max_time)[1] - 1e-6
        title = '$\kappa_{{{}}}$'.format(j)
        plot_1_kappa_paper(x, j, eta, xsample, title, color, ax=axd[key], show=False)

    if save is not None:
        plt.savefig(save, dpi=300)
    if show is True:
        plt.show()

    return

def plot_4_kappas_paper(x, xsample, color, save=None, show=True):
    fig = plt.figure(layout='constrained', figsize=(16, 4))
    mosaic = """
             ABCD
             """
    axd = fig.subplot_mosaic(mosaic) 
    axd_keys = ['A', 'B', 'C', 'D']
    harmonics = [1, -2, 3, -4]
    for key, j in zip(axd_keys, harmonics):
        eta = ha.eta_exclusive_bounds(j, max_time)[1] - 1e-6
        title = '$\kappa_{{{}}}$'.format(j)
        plot_1_kappa_paper(x, j, eta, xsample, title, color, ax=axd[key], show=False)

    if save is not None:
        plt.savefig(save, dpi=300)
    if show is True:
        plt.show()

    return

if __name__=="__main__":
    # Size for lines in plot for the PAPER.
    mpl.rcParams['axes.linewidth'] = 4
    mpl.rcParams['xtick.major.size'] = 7#5
    mpl.rcParams['xtick.major.width'] = 2 
    mpl.rcParams['xtick.minor.size'] = 1
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.major.size'] = 7#5
    mpl.rcParams['ytick.major.width'] = 2 
    mpl.rcParams['ytick.minor.size'] = 1
    mpl.rcParams['ytick.minor.width'] = 1
    plt.rcParams['figure.constrained_layout.use'] = True
    mpl.rcParams['font.sans-serif'] = ['Droid Sans']

    x = np.linspace(0, max_time, 1000)
    c = '#984EA3' # Deep Lilac

    # Make initial sequence PDD of length 6
    Nsample = 4
    xsample = ps.PDD(Nsample, max_time) 

    #plot_3_kappas_paper(x, xsample, c, save='paper_plots/Figure2b.svg')
    #plot_3_kappas_paper(x, xsample, c)
    #plot_4_kappas_paper(x, xsample, c, save='paper_plots/Figure2b.svg')
    plot_4_kappas_paper(x, xsample, c, save='paper_plots/posterfest_action_funcs.png')
     
    #for j in harmonic_list:
    #    title = '$\kappa_{{{}}}$'.format(j)
    #    eta = ha.eta_exclusive_bounds(j, max_time)[1] - 1e-6
    #    plot_1_kappa_paper(x, j, eta, xsample, title, c)
        #plt.savefig(fname, dpi = 300)
        #plt.close(fig=fig)
