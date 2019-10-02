# Adaptation of Method 3 from Hale, Higham, and Trefethen, Computing f(A)b by contour integrals. SIAM 2008
from scipy.special import *


run_test = True # requires numpy


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


if run_test:
    import numpy as np
    m = 1e-1
    M = 1e3
    num_test_pts = 1e5

    zz = np.logspace(np.log10(m), np.log10(M), num_test_pts)
    ff = 1. / np.sqrt(zz)

    for N in range(1, 21):
        ww, pp, rat = rational_inverse_square_root_weights_and_poles(m, M, N)
        ff_rat = rat(zz)

        max_err = np.max(np.abs(ff - ff_rat))
        l2_err = np.linalg.norm(ff - ff_rat)
        print('N=', N, ', max_err=', max_err, ', l2_err=', l2_err)
