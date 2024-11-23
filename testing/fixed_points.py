import numpy as np
import matplotlib.pyplot as plt

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
    ell_max = ( abs(harmonic) - 1 - (abs(harmonic)-1)%2 ) / 2
    ell = np.arange(ell_max + 1)
    m_max = (abs(harmonic) - abs(harmonic) % 2) / 2
    m = np.arange(m_max + 1)
    if harmonic > 0:
        attracting_pts = (2 * ell + 1) * max_time / harmonic
        repelling_pts = 2 * m * max_time / harmonic
    if harmonic < 0:
        attracting_pts = - 2 * m * max_time / harmonic
        repelling_pts = - (2 * ell + 1) * max_time / harmonic
    return attracting_pts, repelling_pts

if __name__=='__main__':
    """
    TODO:
    - Finish testing that these fixed points are correct, both for positive and negative j.
    - Plot bunch of these points. Attracting points are circles. Repelling points are x's.
      The x-axis values are given by the fixed point, and the y axis values are given by the jth harmonic.
      The circles and x's are on a time axis, with 0 and T labelled. All axes should share x-axis. Y-axis line
      should be suppressed, but the j label should be shown. The x-axis labels should be suppressed for all
      but the bottom row.
    """
    j = -1
    maxTime = 1
    attracting_pts, repelling_pts = fixed_points(j, maxTime)
    print("Attracting Points")
    print(attracting_pts)
    print("Repelling Points")
    print(repelling_pts)
