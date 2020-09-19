# Adaptation of Method 3 from Hale, Higham, and Trefethen, Computing f(A)b by contour integrals. SIAM 2008
import numpy as np
from scipy.special import *


def rational_inverse_square_root_weights_and_poles(interval_min_m, interval_max_M, number_of_rational_terms_N):
    # Computes weights and poles for rational approximation to the inverse square root function:
    #     1/sqrt(z) =approx= w0/(z - p0) + w1/(z - p1) + ... + wN/(z - pN)
    # Approximation designed to be as accurate as possible on the interval [m, M]
    m = interval_min_m
    M = interval_max_M
    N = number_of_rational_terms_N
    k2 = m/M
    Kp = ellipk(1-k2)
    t = 1j * np.arange(0.5, N) * Kp/N
    sn, cn, dn, ph = ellipj(t.imag,1-k2)
    cn = 1./cn
    dn = dn * cn
    sn = 1j * sn * cn
    w = np.sqrt(m) * sn
    dzdt = cn * dn

    poles = (w**2).real
    weights = (2 * Kp * np.sqrt(m) / (np.pi*N)) * dzdt
    rational_function = lambda zz: np.dot(1. / (zz.reshape((-1,1)) - poles), weights)
    return weights, poles, rational_function

