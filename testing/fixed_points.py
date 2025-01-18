import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pulse_sequence as ps

def fixed_points(harmonic, max_time):
    """
    Calculate the attracting and repelling fixed points associated with successive
    function compositions of the stretching action function of harmonic number harmonic.
    PARAMETERS:
    harmonic (int): the number of the harmonic we are considering.
                    Must be nonzero, since only action functions
                    with nonzero harmonics have attracting and repelling fixed points.
    max_time (float): the max value of the interval of interest. The interval is
                        [0, max_time].
    RETURNS:
    attracting_pts (np.ndarray of shape (1,)), repelling_pts (np.ndarray of shape (1,)):
                    attracting and repelling fixed points of the 
    """
    if harmonic == 0:
        error_message = """Variable harmonic cannot be zero,
                         since only action functions with nonzero harmonics
                         have attracting and repelling fixed points."""
        raise ValueError(error_message)
    b = np.arange(abs(harmonic) + 1)
    even = b[::2] * max_time / abs(harmonic)
    odd = b[1::2] * max_time / abs(harmonic)
    if harmonic > 0:
        attracting_pts = odd
        repelling_pts = even
    if harmonic < 0:
        attracting_pts = even
        repelling_pts = odd
    return attracting_pts, repelling_pts

def plot_fixed_points(harmonic_array, max_time, color = 'red', show = True, save = False, pulseseq = None):
    """
    Plot fixed points for various action functions.
    
    PARAMETERS:
    harmonic_array (np.ndarray, shape (1,)): array of integers representing the harmonics
                                            of each action function
    max_time (float): the max value of the interval of interest. The interval is
                        [0, max_time].
    color (string): color of the fixed points
    show (bool): flag for whether or not to show the plot using matplotlib GUI
    save (bool or string): if False, do not save figure. If string, save plot with file name save.

    RETURNS:
    None
    """
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['xtick.major.size'] = 6
    mpl.rcParams['xtick.major.width'] = 4 
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['xtick.minor.width'] = 5
    # Define font sizes
    SIZE_DEFAULT = 18
    SIZE_LARGE = 20
    # Define pad
    PAD_DEFAULT = 10

    fig, ax = plt.subplots(layout = 'constrained', figsize = (8, 8))
    marker_size = 150 
    ytick_locs = []
    ytick_labels = []
    for idx, h in np.ndenumerate(harmonic_array):
        attracting_pts, repelling_pts = fixed_points(h, max_time)
        # Plot these points. Attracting points are circles. Repelling points are x's.
        # The y values are the list index of the harmonic in the list.
        ax.scatter(attracting_pts, np.full_like(attracting_pts, idx), marker = 'o', c = color, s = marker_size)
        ax.scatter(repelling_pts, np.full_like(repelling_pts, idx), marker = 'x', c = color, s = marker_size)
        # Make yticks.
        ytick_locs.append(idx[0])
        ytick_labels.append('$j = {}$'.format(h))
    
    # Set tick labels
    ax.set_yticks(ticks = ytick_locs, labels = ytick_labels, fontsize = SIZE_DEFAULT)
    xtick_locs = [0, max_time/4, max_time/2, 3*max_time/4, max_time]
    xtick_labels = ['$0$', '$0.25T$', '$0.5T$', '$0.75T$', '$T$']
    ax.set_xticks(ticks = xtick_locs, labels = xtick_labels, fontsize = SIZE_DEFAULT)

    # Remove y axis ticks
    ax.tick_params(left = False)
    # Remove all border lines except for bottom
    ax.spines[['left', 'top', 'right']].set_visible(False)
   
    
    # Set gridlines
    ax.grid()
    
    if pulseseq is not None:
        ax.vlines(pulseseq, 0, len(harmonic_array)-1)
        ax.set_title('Attracting and Repelling Fixed Points of $\kappa_j(t)$ Overlaid with Pulse Sequence', fontsize = SIZE_LARGE)
    else:
        ax.set_title('Attracting and Repelling Fixed Points of $\kappa_j(t)$', fontsize = SIZE_LARGE)
    if save:
        plt.savefig(save, dpi=300)
    if show:
        plt.show()
    return

if __name__=='__main__':
    """
    TODO:
    """
    maxTime = 1
    #harmonics = np.array([-4, -3, -2, -1, 1, 2, 3, 4]) 
    J = 9
    positive = np.arange(1, J+1)
    harmonics = np.concatenate((np.flip(-positive), positive))
    save = None#'paper_plots/fixed_points.png'
    show = True
    pulseseq = ps.PDD(8, maxTime)
    plot_fixed_points(harmonics, maxTime, save = save, show = show, pulseseq = pulseseq)
