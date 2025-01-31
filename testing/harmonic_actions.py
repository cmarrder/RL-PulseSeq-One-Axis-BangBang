import numpy as np

def kappa(t, harmonic, eta, max_time):
    """
    Calculate the action function kappa.
    PARAMETERS:
    t (np.ndarray): mesh of time values ranging from 0 to max_time.
    harmonic (int): the number of the harmonic we are considering.
                    Must be nonzero, since only action functions
                    with nonzero harmonics have attracting and repelling fixed points.
    eta (float): nonlinearity parameter.
    max_time (float): the max value of the time interval
    RETURNS:
    np.ndarray
    """
    return t + eta * np.sin(np.pi * harmonic * t / max_time)

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

def eta_exclusive_bounds(harmonic, max_time):
    """
    Calculate the exclusive (i.e. noninclusive) bounds on the eta parameter.
    To maintain bijectivity of the action function kappa, eta must lie strictly between
    these bounds.
    PARAMETERS:
    harmonic (int): the number of the harmonic we are considering.
                    Must be nonzero, since only action functions
                    with nonzero harmonics have attracting and repelling fixed points.
    max_time (float): the max value of the interval of interest. The interval is
                        [0, max_time].
    RETURNS:
    tuple
    """
    return (0, max_time / np.pi / abs(harmonic))
