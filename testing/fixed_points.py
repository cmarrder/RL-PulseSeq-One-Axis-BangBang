import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pulse_sequence as ps
import harmonic_actions as ha
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import axes3d

def plot_fixed_points(harmonic_array, max_time, color = 'red', show = True, save = False, pulseseq = None):
    """
    Plot fixed points for various action functions.
    
    PARAMETERS:
    harmonic_array (np.ndarray, shape (1,)): array of integers representing the harmonics
                                            of each action function
    max_time (float): the max value of the interval of interest. The interval is
                        [0, max_time].
    color (Optional, string): color of the fixed points
    show (Optional, bool): flag for whether or not to show the plot using matplotlib GUI
    save (Optional, bool or string): if False, do not save figure. If string, save plot with file name save.
    pulseseq (Optional, 1D np.ndarray): locations of vertical lines to indicate location of pulses

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
        attracting_pts, repelling_pts = ha.fixed_points(h, max_time)
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

def plot_apps_vs_eta(attracting_pt, repelling_pt, harmonic, max_time, fit = True):
    """ Plot fixed point convergence as a function of eta.
    """
    Neta = 100
    max_applications = 1e8

    # Make mesh of etas using its exclusive bounds. Cutoff first and last points because
    # they are noninclusive bounds.
    etas = np.linspace(*ha.eta_exclusive_bounds(harmonic, max_time), Neta+2)[1:Neta+1]

    epsilon = max_time / max_applications # Error tolerance

    # Initialize start and end points on time axis.
    start_pt = repelling_pt + np.sign(attracting_pt - repelling_pt) * epsilon
    end_pt = attracting_pt

    # For each eta, we will track how many times we had to apply kappa function to converge to
    # the attracting fixed point.
    application_counts = np.zeros_like(etas)

    for index, eta in np.ndenumerate(etas):
        applications = 0
        current_pt = start_pt
        # Iteratively apply kappa function, and count how many times we apply it
        # before we get epsilon-close to end_pt.
        while abs(current_pt - end_pt) > epsilon:
            current_pt = ha.kappa(current_pt, harmonic, eta, max_time)
            applications += 1
            if applications > max_applications:
                applications = -max_applications # Negative value will signal we exceeded allowable count limit.
                break # Exit while loop
        application_counts[index] = applications

    plt.scatter(etas, application_counts, label = 'data', color = '#D81B60')
    plt.xlabel('$\eta_{{{}}}$'.format(harmonic))
    plt.ylabel('Number of Times $\kappa_{{{}}}(t)$ Applied'.format(harmonic))
    
    if fit:
        #def exp_decay_fit(x, a, b, c):
        #    return a * np.exp(- x * b) + c
        #exp_fit_param, cov = curve_fit(exp_decay_fit, etas, application_counts)
        #plt.plot(etas, exp_decay_fit(etas, *exp_fit_param), label = 'Exponential Fit', color = '#004D40')

        def power_law_fit(x, a, b, c):
            return a * x**(-b) + c
        power_fit_param, cov = curve_fit(power_law_fit, etas, application_counts)
        print("Power Law Fitting Parameters")
        print(power_fit_param)
        plt.plot(etas, power_law_fit(etas, *power_fit_param), label = 'Power Law Fit', color = '#1E88E5')

        plt.legend()
    
    plt.show()
    return

def frac_fixed_pt_traversed(current_pt, attracting_pt, repelling_pt):
    # Check current_pt between attracting_pt and repelling_pt
    interval = abs(attracting_pt - repelling_pt)
    fraction_traversed = (current_pt + min(attracting_pt, repelling_pt)) / interval
    return fraction_traversed

def plot3d_eta_apps_distance(attracting_pt, repelling_pt, harmonic, max_time, fit = True, show=True):
    """ Plot fixed point convergence as a function of eta.
    """
    Neta = 100
    max_applications = 100

    # Make mesh of etas using its exclusive bounds. Cutoff first and last points because
    # they are noninclusive bounds.
    etas = np.linspace(*ha.eta_exclusive_bounds(harmonic, max_time), Neta+2)[1:Neta+1]

    epsilon = max_time / max_applications # Error tolerance

    # Initialize start and end points on time axis.
    start_pt = repelling_pt + np.sign(attracting_pt - repelling_pt) * epsilon
    end_pt = attracting_pt
    
    current_pts = np.full_like(etas, start_pt) # Array of current points along time axis. Same size as eta.
    applications = np.arange(1, max_applications+1) # Array of function application counts.
    frac_traversed = np.zeros((len(applications), len(current_pts))) # 2D array storing fraction traversed data

    # Apply the kappa function to the current points repeatedly, each time recording their fraction traversed.
    for i in range(max_applications):
        current_pts = ha.kappa(current_pts, harmonic, etas, max_time)
        frac_traversed[i] = frac_fixed_pt_traversed(current_pts, attracting_pt, repelling_pt) # Check this indexing works!

    # Make meshgrids for the independent variables
    eta_mg, app_mg = np.meshgrid(etas, applications)
    

    # Surface plot code here:
    ax = plt.figure().add_subplot(projection='3d')
    # Plot the 3D surface
    #ax.plot_surface(eta_mg, app_mg, frac_traversed, cmap='coolwarm', antialiased=True)
    ax.plot_surface(eta_mg, app_mg, frac_traversed, cmap='coolwarm', edgecolor='black', lw=0.25, antialiased=True)
    #ax.plot_surface(eta_mg, app_mg, frac_traversed, cmap='coolwarm', lw=3.5, rstride=5, cstride=5)
    #ax.plot_surface(eta_mg, app_mg, frac_traversed, edgecolor='royalblue', lw=0.5, rstride=5, cstride=5,
    #                alpha=0.3)

    xlim = (0, 1.25 * etas[-1])
    ylim = (0, 1.25 * applications[-1])
    xlabel = '$\eta_{{{}}}$'.format(harmonic)
    ylabel = 'M'
    zlabel = '$\gamma_{{{}}}$'.format(harmonic)
    
    #ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    #ax.contour(eta_mg, app_mg, frac_traversed, zdir='x', offset=xlim[1], cmap='coolwarm')
    #ax.contour(eta_mg, app_mg, frac_traversed, zdir='y', offset=ylim[1], cmap='coolwarm')
    
    ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    
    ax.grid(False)
    if fit:

        # First need flatten data
        # HERE
        # Now define model function
        def power_law_fit(X, a, b):
            x, y = X
            return a * x**(b) * y
        fit_param, fit_cov = curve_fit(power_law_fit, (eta_mg.flatten(), app_mg.flatten()), frac_traversed.flatten())
        fit_err = np.sqrt(np.diag(fit_cov))
        print("Power Law Fitting Parameters")
        print(fit_param)
        print("Fitting Error")
        print(fit_err)

        ax.scatter(eta_mg.flatten(), app_mg.flatten(), frac_traversed.flatten(), s=1)
    if show:
        plt.show()
    else:
        plt.close()
    if fit:
        return fit_param, fit_err
    return

def average_change(attracting_pt, repelling_pt, harmonic, eta, max_time):
    t = np.linspace(min(attracting_pt, repelling_pt), max(attracting_pt, repelling_pt), 1000)
    dt = t[1]-t[0]
    kappa = ha.kappa(t, harmonic, eta, max_time)
    return np.abs(np.sum(kappa - t) * dt / (attracting_pt - repelling_pt))

if __name__=='__main__':
    maxTime = 1
    #harmonics = np.array([-4, -3, -2, -1, 1, 2, 3, 4]) 
    J = 1
    positive = np.arange(1, J+1)
    harmonics = np.concatenate((np.flip(-positive), positive))
    apts, rpts = ha.fixed_points(J, maxTime)

    save = None#'paper_plots/fixed_points.png'
    show = True
    pulseseq = ps.PDD(8, maxTime)

    # Average change across fixed point interval
    test_eta = ha.eta_exclusive_bounds(J, maxTime)[1]/2
    print("Numerical avg: ", average_change(apts[0], rpts[0], J, test_eta, maxTime))
    print("Exact_avg: ", 2*test_eta/np.pi)

    # Plot Fixed Points
    #plot_fixed_points(harmonics, maxTime, save = save, show = show, pulseseq = pulseseq)

    # Plot Convergence to Fixed Points
    #plot_apps_vs_eta(apts[0], rpts[0], J, maxTime)
    plot3d_eta_apps_distance(apts[0], rpts[0], J, maxTime, fit=True)

    """
    Jmax = 20
    fit_param_arr = np.zeros((Jmax, 2)) 
    fit_err_arr = np.zeros((Jmax, 2)) 
    for j in range(1, Jmax+1):
        fit_param_arr[j-1], fit_err_arr[j-1] = plot3d_eta_apps_distance(apts[0], rpts[0], j, maxTime, show=False)
    param1 = fit_param_arr[:, 0]
    param2 = fit_param_arr[:, 1]
    param1_err = fit_err_arr[:, 0]
    param2_err = fit_err_arr[:, 1]

    plt.errorbar(param1, param2, xerr=param1_err, yerr=param2_err)
    plt.show()
    """
