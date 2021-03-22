import numpy as np
from scipy.signal import convolve


def conforming_grid_convolution(F1, min1, max1, F2, min2, max2, p1=None, p2=None, run_checks=True, method='auto'):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/blob/master/jupyter_notebooks/conforming_grid_convolution.ipynb
    d = len(F1.shape)

    if p1 is None:
        p1 = np.zeros(d)
    if p2 is None:
        p2 = np.zeros(d)

    hh = (max1 - min1) / (np.array(F1.shape) - 1)

    if run_checks:
        hh2 = (max2 - min2) / (np.array(F2.shape) - 1)
        if np.linalg.norm(hh - hh2) > 1e-10:
            raise RuntimeError('Grids not conforming (different spacings h)')

        if (not is_divisible_by(min2 - min1, hh)):
            raise RuntimeError('Grids not conforming (one grid is shifted relative to the other)')

    element_volume = np.prod(hh)

    F3 = convolve(F1, F2, mode='full', method=method) * element_volume
    min3 = (min1 - p1) + (min2 - p2)
    max3 = (max1 - p1) + (max2 - p2)
    return F3, min3, max3


def is_divisible_by(xx, yy, tol=1e-10):
    return np.linalg.norm(xx / yy - np.rint(xx / yy)) < tol
