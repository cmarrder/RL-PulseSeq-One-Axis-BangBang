import numpy as np
import matplotlib.pyplot as plt
import pulse_sequence as ps

def legendreQuad5(xmin, xmax, N_interval):
    """
    Sample points and weights for Gauss-Legendre quadrature.
    PARAMETERS:
    xmin (float): minimum value of abscissa
    xmax (float): maximum value of abscissa
    N_interval (int): number of intervals to break up region [xmin, xmax]
    RETURNS:
    points (np.ndarray, shape (5*N_interval,)), weights (np.ndarray, shape (5*N_interval,))
    """
    # Initialize the Gaussian points and weights
    # on the bounded base interval [-1, 1].
    bpoints = np.zeros(5)
    bweights = np.zeros(5)

    bpoints[0] = - np.sqrt(5 + 2 * np.sqrt(10/7)) / 3
    bpoints[1] = - np.sqrt(5 - 2 * np.sqrt(10/7)) / 3
    bpoints[3] = - bpoints[1]
    bpoints[4] = - bpoints[0]

    bweights[0] = (322 - 13 * np.sqrt(70)) / 900
    bweights[1] = (322 + 13 * np.sqrt(70)) / 900
    bweights[2] = 128/225
    bweights[3] = bweights[1]
    bweights[4] = bweights[0]

    # Break the integrate interval [xmin, xmax] into N_interval intervals
    dx = (xmax - xmin) / N_interval

    points = np.ones(5*N_interval)
    weights = np.ones(5*N_interval)

    x = xmin + np.arange(N_interval+1) * dx

    half_diff = (x[1:N_interval+1] - x[0:N_interval]) / 2
    midpoints = (x[1:N_interval+1] + x[0:N_interval]) / 2

    for k in range(N_interval):
        weights[5*k : 5*(k+1)] = half_diff[k] * bweights
        points[5*k : 5*(k+1)] = half_diff[k] * bpoints + midpoints[k]
    return points, weights

def chebyshevQuad1stKind(xmin, xmax, degree_N):
    """ Sample points and weights for Chebyshev-Gauss quadrature.
    PARAMETERS:
    xmin (float): minimum value of abscissa
    xmax (float): maximum value of abscissa
    degree_N (int): degree of polynomial of Chebyshev polynomials of the first kind
    RETURNS
    points (np.ndarray, shape (degree_N,)), weights (np.ndarray, shape (degree_N,))
    """
    indices = np.arange(degree_N)
    half_diff = (xmax - xmin) / 2
    midpoint = (xmin + xmax) / 2
    bpoints = np.cos( (2 * indices + 1) * np.pi / 2 / degree_N ) # Points on the base interval [-1, 1]
    points = half_diff * bpoints + midpoint
    #weights = np.full_like(points, 0.5 * (xmax - xmin) * (np.pi / degree_N), dtype='float')
    #weights = half_diff * np.sqrt(1 - (half_diff * bpoints + midpoint)**2) * np.pi / degree_N
    weights = half_diff * np.sqrt(1 - bpoints**2) * np.pi / degree_N

    return points, weights

def plotChebyshevNodes1stKind(xmin, xmax, degree_N, func=None, yscale='linear'):
    x, _ = chebyshevQuad1stKind(xmin, xmax, degree_N)
    if func is not None:
        plt.vlines(x, 0, func(x))
        plt.ylabel('$f(x)$')
        plt.yscale(yscale)
    else:
        plt.vlines(x, 0, 1)

    plt.xlabel('$x$')
    plt.title('Chebyshev Nodes of Degree {}'.format(degree_N))
    plt.show()
    return

def plotLegendreNodesDegree5(xmin, xmax, N_interval, func=None, yscale='linear'):
    x, _ = legendreQuad5(xmin, xmax, N_interval)
    if func is not None:
        plt.vlines(x, 0, func(x))
        plt.ylabel('$f(x)$')
        plt.yscale(yscale)
    else:
        plt.vlines(x, 0, 1)

    plt.xlabel('$x$')
    plt.title('Legendre Nodes of Degree 5 over {} Intervals'.format(N_interval))
    plt.show()
    return

if __name__=='__main__':
    import time
    # Plot accuracy of free induction decay filter
    plot_free_ind_decay_err = False
    # Plot UDD and CPMG filter functions alongside noise
    plot_UDD_CPMG = False
    # Plot accuracy of single random pulse sequence
    plot_single_random = False
    # Plot average, and standard deviation for a number of random pulse sequences
    # as a function of number of gaussian quadrature intervals used
    plot_many_random = True
    
    nPulse = 8
    tMax = 1
    nTimeStep = 2000
    dTime = tMax / nTimeStep
    nyquistFreq = 0.5 / dTime
    SFunc = lambda x: 1 / x # ps.one_over_f_noise
    
    # Frequencies and noise to be used in rectangular method
    nFreqRec = 100000
    tempFreqRec = np.linspace(0, nyquistFreq, nFreqRec)
    tempSRec = ps.one_over_f_noise(tempFreqRec)
    freqMax = tempFreqRec[ps.cutoff_index(tempFreqRec, tempSRec)]
    freqMin = 1/tMax#0
    freqRec = np.linspace(freqMin, freqMax, nFreqRec) 
    SRec = SFunc(freqRec)
    print('freqMax:', freqMax)

    if plot_free_ind_decay_err:
        t0 = time.time()
        # Frequencies and noise to be used in Gaussian quadrature method
        min_nInterval = 5
        max_nInterval = 3000 
        arr_nInterval = np.arange(min_nInterval, max_nInterval + 1)
        arr_avg_chi_err = np.zeros_like(arr_nInterval, dtype='float')
        arr_std_chi_err = np.zeros_like(arr_nInterval, dtype='float')

        num_iterations = len(arr_nInterval)
        counter = 0
        ffRec = ps.FIDFilter(freqRec, tMax)
        chiRec = ps.chi(freqRec, SRec, ffRec)
        arr_chi_err = np.zeros_like(arr_nInterval, dtype='float')
        chiExact = 0.02256066174635 * tMax**2
        for i, nInterval in np.ndenumerate(arr_nInterval):
            # Gaussian quadrature points and weights
            freqChebyshev, weightChebyshev = chebyshevQuad1stKind(freqMin, 1, 500)
            freqLegendre, weightLegendre = legendreQuad5(1, freqMax, nInterval)
            freqGQ = np.concatenate((freqChebyshev, freqLegendre))
            weights = np.concatenate((weightChebyshev, weightLegendre))
            # Make noise spectrum
            SGQ = SFunc(freqGQ)
            # Initialize
            ffGQ = ps.FIDFilter(freqGQ, tMax)
            chiGQ = ps.chi(freqGQ, SGQ, ffGQ, weights=weights)
            arr_chi_err[i] = np.abs(chiGQ - chiExact) / chiExact # Compare to exact expression for chi
            #arr_chi_err[i] = np.abs(chiGQ - chiRec) / chiRec # Compare to numerical expression
            counter += 1
            if (counter % int(num_iterations/100)) == 0:
                print('Percent completed: {}'.format(counter / num_iterations * 100))
        t1 = time.time()
        print('Time elapsed: ', t1-t0)

        print(np.sum(SFunc(freqChebyshev) * ps.FIDFilter(freqChebyshev, tMax) * weightChebyshev))
        print(np.sum(SFunc(freqLegendre) * ps.FIDFilter(freqLegendre, tMax) * weightLegendre))
        print('chiGQ: ', chiGQ)
        print('chiRec: ', chiRec)
        print('chiExact: ', chiExact)

        plt.plot(arr_nInterval, arr_chi_err)
        plt.xlabel('Number of intervals used in Gaussian quadrature')
        plt.ylabel('$|\chi_{gq} - \chi_{rec}| / \chi_{rec}$')
        plt.yscale('log')
        plt.show()

        plt.vlines(freqGQ, 0, SGQ * ffGQ)
        plt.show()


    if plot_single_random:
        t0 = time.time()
        # Frequencies and noise to be used in Gaussian quadrature method
        min_nInterval = 5
        max_nInterval = 500 
        arr_nInterval = np.arange(min_nInterval, max_nInterval + 1)
        arr_avg_chi_err = np.zeros_like(arr_nInterval, dtype='float')
        arr_std_chi_err = np.zeros_like(arr_nInterval, dtype='float')
        pulse_times = ps.randomSequence(nPulse, tMax)

        num_iterations = len(arr_nInterval)
        counter = 0
        ffRec = ps.FilterFunc(freqRec, pulse_times, tMax)
        chiRec = ps.chi(freqRec, SRec, ffRec)
        arr_chi_err = np.zeros_like(arr_nInterval, dtype='float')
        for i, nInterval in np.ndenumerate(arr_nInterval):
            # Gaussian quadrature points and weights
            freqGQ, weights = legendreQuad5(freqMin, freqMax, nInterval)
            # Make noise spectrum
            SGQ = SFunc(freqGQ)
            # Initialize
            ffGQ = ps.FilterFunc(freqGQ, pulse_times, tMax)
            chiGQ = ps.chi(freqGQ, SGQ, ffGQ, weights=weights)
            arr_chi_err[i] = np.abs(chiGQ - chiRec) / chiRec
            counter += 1
            if (counter % int(num_iterations/100)) == 0:
                print('Percent completed: {}'.format(counter / num_iterations * 100))
        t1 = time.time()
        print('Time elapsed: ', t1-t0)

        print('chiGQ: ', chiGQ)
        print('chiRec: ', chiRec)

        plt.plot(arr_nInterval, arr_chi_err)
        plt.xlabel('Number of intervals used in Gaussian quadrature')
        plt.ylabel('$|\chi_{gq} - \chi_{rec}| / \chi_{rec}$')
        plt.yscale('log')
        plt.show()


    if plot_many_random:
        t0 = time.time()
        # Frequencies and noise to be used in Gaussian quadrature method
        min_nInterval = 5
        max_nInterval = 200 
        arr_nInterval = np.arange(min_nInterval, max_nInterval + 1)
        nSeq = 100
        arr_avg_chi_err = np.zeros_like(arr_nInterval, dtype='float')
        arr_std_chi_err = np.zeros_like(arr_nInterval, dtype='float')
        pulse_times = ps.randomSequence(nPulse, tMax, nSequence=nSeq) 

        num_iterations = len(arr_nInterval) * nSeq
        counter = 0
        t0 = time.time()
        for i, nInterval in np.ndenumerate(arr_nInterval):
            # Gaussian quadrature points and weights
            freqGQ, weights = legendreQuad5(freqMin, freqMax, nInterval)
            # Make noise spectrum
            SGQ = ps.one_over_f_noise(freqGQ)
            # Initialize
            arr_chi_err = np.zeros(nSeq)
            for j in range(nSeq):
                ffGQ = ps.FilterFunc(freqGQ, pulse_times[j], tMax)
                ffRec = ps.FilterFunc(freqRec, pulse_times[j], tMax)
                chiGQ = ps.chi(freqGQ, SGQ, ffGQ, weights=weights)
                chiRec = ps.chi(freqRec, SRec, ffRec)
                arr_chi_err[j] = np.abs(chiGQ - chiRec) / chiRec
                counter += 1
                if (counter % int(num_iterations/100)) == 0:
                    print('Percent completed: {}'.format(counter / num_iterations * 100))
            arr_avg_chi_err[i] = np.mean(arr_chi_err)
            arr_std_chi_err[i] = np.std(arr_chi_err)
        t1 = time.time()
        print('Time elapsed: ', t1-t0)

        plt.errorbar(arr_nInterval, arr_avg_chi_err, yerr=arr_std_chi_err)
        plt.xlabel('Number of intervals used in Gaussian quadrature')
        plt.ylabel('Average $|\chi_{gq} - \chi_{rec}|$')
        plt.show()

    
    if plot_UDD_CPMG:
        t0 = time.time()
        # Frequencies and noise to be used in Gaussian quadrature method
        nInterval = 100 
        freqGQ, weights = legendreQuad5(freqMin, freqMax, nInterval)
        SGQ = ps.one_over_f_noise(freqGQ)
        
        # Crunch numbers
        UDDTime = ps.UDD(nPulse, tMax)
        CPMGTime = ps.CPMG(nPulse, tMax)
        
        # Filters to be used in rectangular method
        UDDFilterRec = ps.FilterFunc(freqRec, UDDTime, tMax)
        CPMGFilterRec = ps.FilterFunc(freqRec, CPMGTime, tMax)
        
        # Filters to be used in Gaussian quadrature method
        UDDFilterGQ = ps.FilterFunc(freqGQ, UDDTime, tMax)
        CPMGFilterGQ = ps.FilterFunc(freqGQ, CPMGTime, tMax)
        
        # Compare overlaps
        dfreq = freqRec[1] - freqRec[0]
        UDDOverlapRec = np.sum(SRec * UDDFilterRec * dfreq)
        CPMGOverlapRec = np.sum(SRec * CPMGFilterRec * dfreq)
        
        UDDOverlapGQ = np.sum(SGQ * UDDFilterGQ * weights)
        CPMGOverlapGQ = np.sum(SGQ * CPMGFilterGQ * weights)
        
        UDDColor = 'blue'
        CPMGColor = 'red'
        
        UDDRecLabel = '$\chi_{{UDD}}^{{rec}}$ = {:.4e}'.format(UDDOverlapRec)
        CPMGRecLabel ='$\chi_{{CPMG}}^{{rec}}$ = {:.4e}'.format(CPMGOverlapRec)
        
        UDDGQLabel = '$\chi_{{UDD}}^{{gq}}$ = {:.4e}'.format(UDDOverlapGQ)
        CPMGGQLabel ='$\chi_{{CPMG}}^{{gq}}$ = {:.4e}'.format(CPMGOverlapGQ)
        
        fig, axs = plt.subplots(3, sharex=True)
        
        axs[0].set_title('$N_{{pulse}} = {}$'.format(nPulse))
        axs[0].plot(freqRec, SRec)
        axs[0].set_ylabel(r"$|S(\nu)|^2$")
        
        axs[1].plot(freqRec, UDDFilterRec, color = UDDColor, label = "UDD")
        axs[1].plot(freqRec, CPMGFilterRec, color = CPMGColor, label = "CPMG")
        axs[1].set_ylabel(r"$F(\nu)$")
        axs[1].legend()
        
        axs[2].plot(freqRec, SRec*UDDFilterRec, color = UDDColor, label = UDDRecLabel)
        axs[2].scatter(freqGQ, SGQ*UDDFilterGQ, color = UDDColor, label = UDDGQLabel)
        axs[2].plot(freqRec, SRec*CPMGFilterRec, color = CPMGColor, label = CPMGRecLabel)
        axs[2].scatter(freqGQ, SGQ*CPMGFilterGQ, color = CPMGColor, label = CPMGGQLabel)
        axs[2].set_ylabel(r"$F(\nu) * S|(\nu)|^2$")
        axs[2].legend()
        
        plt.show()
        
        print("UDD")
        print("Rectangle Method: {}".format(UDDOverlapRec))
        print("Gaussian Quadrature Method: {}".format(UDDOverlapGQ))
        
        print("CPMG")
        print("Rectangle Method: {}".format(CPMGOverlapRec))
        print("Gaussian Quadrature Method: {}".format(CPMGOverlapGQ))
