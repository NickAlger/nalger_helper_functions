import numpy as np
import scipy.optimize as optimize
import scipy.integrate as integrate
from math import factorial


run_tests = True # test of correctness and demonstration of usage


#######################    Primary functions    #######################


def vector_polynomial_arc_length_finder(desired_arc_length_S, coefficient_vectors_uu, method='newton', verbose=False):
    # f(t) = uu[:,0] + t * uu[:,1] + t**2 * uu[:,2] + t**3 * uu[:,3] + ...
    # Finds T such that S equals the arc length of the curve f(t) traced from 0 to T
    # Can also use Halley's method
    S = desired_arc_length_S
    uu = coefficient_vectors_uu
    f = lambda t: vector_polynomial_arc_length(t, uu) - S
    g = lambda t: vector_polynomial_arc_length_derivative(t, uu)
    h = lambda t: vector_polynomial_arc_length_second_derivative(t, uu)

    rr = optimize.root_scalar(f, method=method, bracket=[0.0, np.inf], fprime=g, fprime2=h, x0=0.0)
    if verbose:
        print(rr)

    return rr.root


def taylor_arc_length_finder(desired_arc_length_S, taylor_vectors_vv, method='newton', verbose=False):
    # f(t) = vv[:,0] + t * vv[:,1] + (1./2.) * t**2 * vv[:,2] + (1./6.) * t**3 * vv[:,3] + ...
    # Finds T such that S equals the arc length of the curve f(t) traced from 0 to T
    uu = make_coefficient_vectors_from_taylor_vectors(taylor_vectors_vv)
    return vector_polynomial_arc_length_finder(desired_arc_length_S, uu, method=method, verbose=verbose)


#######################    Helper functions    #######################


def eval_vector_polynomial(t, uu):
    zero_dotdot_p = np.arange(uu.shape[1])
    t_to_the_k = (t**zero_dotdot_p).reshape((1,-1))
    ut = np.sum(t_to_the_k * np.array(uu), axis=1)
    return ut


def eval_vector_polynomial_derivative(t, uu):
    zero_dotdot_p = np.arange(uu.shape[1])
    t_to_the_k = (t ** zero_dotdot_p)
    dcc = (zero_dotdot_p[1:] * t_to_the_k[:-1]).reshape((1,-1))
    ut_dot = np.sum(dcc * np.array(uu[:,1:]), axis=1)
    return ut_dot


def eval_vector_polynomial_second_derivative(t, uu):
    zero_dotdot_p = np.arange(uu.shape[1])
    t_to_the_k = (t ** zero_dotdot_p)
    dcc = (zero_dotdot_p[2:] * zero_dotdot_p[1:-1] * t_to_the_k[:-2]).reshape((1, -1))
    ut_double_dot = np.sum(dcc * np.array(uu[:, 2:]), axis=1)
    return ut_double_dot


def vector_polynomial_arc_length_integrand(t, uu):
    u_dot = eval_vector_polynomial_derivative(t, uu)
    return np.linalg.norm(u_dot)


def vector_polynomial_arc_length(T, uu):
    return integrate.quad(lambda t: vector_polynomial_arc_length_integrand(t, uu), 0, T)[0]


def vector_polynomial_arc_length_derivative(T, uu):
    return vector_polynomial_arc_length_integrand(T, uu)


def vector_polynomial_arc_length_second_derivative(T, uu):
    u_dot = eval_vector_polynomial_derivative(T, uu)
    u_doubledot = eval_vector_polynomial_second_derivative(T, uu)
    return np.dot(u_dot, u_doubledot)/np.linalg.norm(u_dot)


#######################    Helper functions for Taylor series    #######################


def vectorized_factorial(xx):
    return np.array([factorial(x) for x in xx])


def make_coefficient_vectors_from_taylor_vectors(vv):
    zero_dotdot_p = np.arange(vv.shape[1])
    one_over_k_factorial = (1. / vectorized_factorial(zero_dotdot_p)).reshape((1, -1))
    uu = one_over_k_factorial * vv
    return uu


def eval_taylor_series(t, vv):
    return eval_vector_polynomial(t, make_coefficient_vectors_from_taylor_vectors(vv))


def taylor_series_arc_length(T, vv):
    return vector_polynomial_arc_length(T, make_coefficient_vectors_from_taylor_vectors(vv))


#######################    Tests    #######################


if run_tests:
    n = 1000
    num_terms = 4
    uu = np.random.randn(n, num_terms)
    t0 = np.random.rand()

    dt = np.random.rand()
    s = 1e-6
    t1 = t0 + s * dt

    u0 = eval_vector_polynomial(t0, uu)
    u1 = eval_vector_polynomial(t1, uu)

    dv_diff = (u1 - u0) / s
    g0 = eval_vector_polynomial_derivative(t0, uu)
    du0 = g0 * dt
    err_vector_polynomial_derivative = np.linalg.norm(dv_diff - du0) / np.linalg.norm(dv_diff)
    print('s=', s, ', err_vector_polynomial_derivative=', err_vector_polynomial_derivative)

    g1 = eval_vector_polynomial_derivative(t1, uu)
    dg_diff = (g1 - g0)/s

    dg = eval_vector_polynomial_second_derivative(t0, uu) * dt
    err_vector_polynomial_second_derivative = np.linalg.norm(dg_diff - dg) / np.linalg.norm(dg_diff)
    print('s=', s, ', err_vector_polynomial_second_derivative=', err_vector_polynomial_second_derivative)

    I0 = vector_polynomial_arc_length(t0, uu)
    I1 = vector_polynomial_arc_length(t1, uu)
    gI0 = vector_polynomial_arc_length_derivative(t0, uu)

    dI_diff = (I1 - I0)/s
    dI = gI0*dt
    err_vector_polynomial_arc_length_derivative = np.abs(dI_diff - dI) / np.abs(dI_diff)
    print('err_vector_polynomial_arc_length_derivative=', err_vector_polynomial_arc_length_derivative)

    gI1 = vector_polynomial_arc_length_derivative(t1, uu)
    dgI_diff = (gI1 - gI0)/s
    dgI = vector_polynomial_arc_length_second_derivative(t0, uu) * dt
    err_vector_polynomial_arc_length_second_derivative = np.abs(dgI_diff - dgI)/np.abs(dgI_diff)
    print('err_vector_polynomial_arc_length_second_derivative=', err_vector_polynomial_arc_length_second_derivative)

    S = np.random.rand()
    T = vector_polynomial_arc_length_finder(S, uu)
    S_r = vector_polynomial_arc_length(T, uu)
    err_arc_length_finder = np.abs(S - S_r) / np.abs(S)
    print('err_arc_length_finder=', err_arc_length_finder)

    T_taylor = taylor_arc_length_finder(S, uu)
    S_r_taylor = taylor_series_arc_length(T_taylor, uu)
    err_taylor_arc_length_finder = np.abs(S - S_r_taylor) / np.abs(S)
    print('err_taylor_arc_length_finder=', err_taylor_arc_length_finder)
