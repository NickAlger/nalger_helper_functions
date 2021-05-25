import numpy as np
from scipy.signal import fftconvolve, convolve
from scipy.special import factorial

import scipy.optimize as optimize
import scipy.integrate as integrate
from scipy.interpolate import pade
from scipy.integrate import quad
from scipy.optimize import root_scalar


class VectorPolynomial:
    def __init__(me, coeffs):
        if isinstance(coeffs, VectorPolynomial):
            coeffs = coeffs.coeffs

        me.coeffs = coeffs
        me.N, me.k = coeffs.shape # coeffs.shape = (spatial_dimension, polynomial_order+1)

    def __call__(me, tt):
        tt = np.array(tt).reshape(-1)
        Nt = len(tt)
        vandermonde = tt.reshape((1, Nt)) ** (np.arange(me.k).reshape((me.k, 1)))  # shape=(k,Nt)
        return np.dot(me.coeffs, vandermonde).T  # shape=(Nt,N)

    def __mul__(me, other):
        if isinstance(other, VectorPolynomial):
            return VectorPolynomial(fftconvolve(me.coeffs, other.coeffs, axes=[1]))
        elif isinstance(other, VectorRationalFunction):
            return other * me # defer to VectorRationalFunction.__mul__()
        else:
            return VectorPolynomial(other * me.coeffs)

    def __rmul__(me, other):
        return me * other

    def __add__(me, other):
        if isinstance(other, VectorPolynomial):
            kmax = np.max([me.k, other.k])

            pp = np.zeros((me.N, kmax))
            pp[:, :me.coeffs.shape[1]] = me.coeffs

            qq = np.zeros((other.N, kmax))
            qq[:, :other.coeffs.shape[1]] = other.coeffs

            return VectorPolynomial(pp + qq)
        elif isinstance(other, VectorRationalFunction):
            return other + me  # defer to VectorRationalFunction.__add__()
        else:
            return VectorPolynomial(me.coeffs + other)

    def __radd__(me, other):
        return me + other

    def __neg__(me):
        return VectorPolynomial(-me.coeffs)

    def __sub__(me, other):
        return me + (-other)

    def __rsub__(me, other):
        return other + (-me)

    def __truediv__(me, other):
        if isinstance(other, VectorPolynomial):
            return VectorRationalFunction(me, other)
        elif isinstance(other, VectorRationalFunction):
            return me.as_rational_function() / other
        else:
            return VectorPolynomial(me.coeffs / other)

    def inv(me):
        return me.as_rational_function().inv()

    def derivative(me):
        dcoeffs = me.coeffs[:, 1:] * np.arange(1, me.k).reshape((1, -1))  # [pp[:,1], ..., pp[:,k]] * [[1,2,3,...,k]]
        return VectorPolynomial(dcoeffs)

    def arc_length(me, T, T0=0.0, display_err=False, tangent=None):
        if tangent is None:
            tangent = me.derivative()
        ds = lambda t: np.linalg.norm(tangent(t))
        L, err = quad(ds, T0, T)
        if display_err:
            print('err=', err)
        return L

    def arc_time(me, L, L0=0.0, display_soln=False, tangent=None, **kwargs):
        if me.k == 1: # curve is constant
            return np.inf

        if tangent is None:
            tangent = me.derivative()

        tangent_length = np.linalg.norm(tangent(L0))

        res = lambda T: L - me.arc_length(T, tangent=tangent)
        soln = root_scalar(res, x0=0.0, x1=0.5*(L-L0)/tangent_length, **kwargs)
        if display_soln:
            print(soln)
        return soln.root

    def as_rational_function(me):
        numerator = me
        # denominator = VectorPolynomial(np.ones((me.N, 1)))
        denominator = ones_vector_polynomial(me.N)
        return VectorRationalFunction(numerator, denominator)

    def __eq__(me, other):
        return (me.coeffs == other.coeffs)

    def __hash__(me):
        return hash(me.coeffs)


class VectorRationalFunction:
    def __init__(me, numerator, denominator):
        me.numerator = VectorPolynomial(numerator)
        me.denominator = VectorPolynomial(denominator)

        me.N = me.numerator.N

    def __call__(me, tt):
        return me.numerator(tt) / me.denominator(tt)

    def __add__(A, B):
        if isinstance(B, VectorRationalFunction):
            C_numerator = A.numerator*B.denominator + A.denominator*B.numerator
            C_denominator = A.denominator * B.denominator
            return C_numerator / C_denominator
        elif isinstance(B, VectorPolynomial):
            return A + B.as_rational_function()
        else:
            return (A.numerator + B) / A.denominator

    def __radd__(A, B):
        return A + B

    def __mul__(A, B):
        if isinstance(B, VectorRationalFunction):
            C_numerator = A.numerator * B.numerator
            C_denominator = A.denominator * B.denominator
            return C_numerator / C_denominator
        elif isinstance(B, VectorPolynomial):
            return A * (B.as_rational_function())
        else:
            return (A.numerator * B) / A.denominator

    def __rmul__(A, B):
        return A * B

    def __neg__(me, other):
        return (-me.numerator) / me.denominator

    def __sub__(A, B):
        return A + (-B)

    def __rsub__(A, B):
        return B + (-A)

    def inv(me):
        return me.denominator / me.numerator

    def __truediv__(A, B):
        if isinstance(B, VectorRationalFunction):
            return A * (B.inv())
        elif isinstance(B, VectorPolynomial):
            return A / (B.as_rational_function())
        else:
            return (A.numerator / B) / A.denominator

    def derivative(A):
        dA_numerator = A.numerator.derivative() * A.denominator - A.denominator.derivative() * A.numerator
        dA_denominator = A.denominator * A.denominator
        return dA_numerator / dA_denominator

    def arc_length(me, T, T0=0.0, display_err=False, tangent=None):
        if tangent is None:
            tangent = me.derivative()
        ds = lambda t: np.linalg.norm(tangent(t))
        L, err = quad(ds, T0, T)
        if display_err:
            print('err=', err)
        return L

    def arc_time(me, L, L0=0.0, display_soln=False, tangent=None, **kwargs):
        if (me.numerator.k == 1) and (me.denominator.k == 1): # curve is constant
            return np.inf

        if tangent is None:
            tangent = me.derivative()

        tangent_length = np.linalg.norm(tangent(L0))

        res = lambda T: L - me.arc_length(T, tangent=tangent)
        soln = root_scalar(res, x0=0.0, x1=0.5*(L-L0)/tangent_length, **kwargs)
        if display_soln:
            print(soln)
        return soln.root

    def as_rational_function(me):
        return me

    def __eq__(me, other):
        return ((me.numerator == other.numerator) and
                (me.denominator == other.denominator))

    def __hash__(me):
        return hash((me.numerator, me.denominator))


def zeros_vector_polynomial(spatial_dimension):
    return VectorPolynomial(np.zeros((spatial_dimension,1)))


def ones_vector_polynomial(spatial_dimension):
    return VectorPolynomial(np.ones((spatial_dimension,1)))


def taylor_vector_polynomial(dvv):
    return VectorPolynomial((1./factorial(np.arange(dvv.shape[1]))).reshape((1,-1)) * dvv)


def pade_vector_rational_function(taylor_polynomial, pade_power, **kwargs):
    taylor_polynomial = VectorPolynomial(taylor_polynomial)  # shape = (N, taylor_order)
    pp = list()
    qq = list()
    for ii in range(taylor_polynomial.N):
        pii, qii = pade(taylor_polynomial.coeffs[ii, :], pade_power, **kwargs)
        pp.append(pii.coeffs[::-1])
        qq.append(qii.coeffs[::-1])
    numerator = VectorPolynomial(np.array(pp))
    denominator = VectorPolynomial(np.array(qq))
    return numerator / denominator


# def polynomial_from_roots_vectorized(roots):
#     spatial_dimension, num_roots = roots.shape
#     one_colvec = np.ones((spatial_dimension,1))
#     poly = VectorPolynomial(one_colvec)
#     for k in range(num_roots):
#         coeffs = np.bmat([-roots[:,k].reshape((-1,1)), one_colvec]) # (x - root)
#         poly = poly * VectorPolynomial(coeffs)
#     return poly
#
#
# def lagrange_polynomial_vectorized(tt, j, spatial_dimension):
#     # tt[i] is the interpolation point for the i'th component of the polynomial.
#     # j is the index of the interpolation point at which the polynomial equals 1.
#     # At all other interpolation points (corresponding to indices i notequal j) the polynomial equals zero.
#     num_pts = len(tt)
#     tt_before = tt[:j]
#     tj = tt[j]
#     tt_after = tt[j+1:]
#     roots = np.concatenate([tt_before, tt_after])
#     roots_matrix = np.outer(np.ones(spatial_dimension), roots)
#     numerator_poly = polynomial_from_roots_vectorized(roots_matrix)
#     denominator = np.prod(tj - roots_matrix, axis=1).reshape((-1,1))
#     lagrange_poly = numerator_poly / denominator
#     return lagrange_poly


def vector_polynomial_interpolation(tt, yy):
    # len(tt) = num_pts
    # yy.shape = (num_pts, spatial_dimension)
    vandermonde = tt.reshape((-1,1)) ** np.arange(len(tt)).reshape((1,-1))
    coeffs = np.linalg.solve(vandermonde, yy).T
    return VectorPolynomial(coeffs)


def polynomial_coeffs_from_roots(rr):
    # (x - rr[0]) * ... * (x - rr[n]) = cc[0] + cc[1] * x + cc[2] * x^2 + ... + cc[n] * x^n
    cc = np.array([1])
    for k in range(len(rr)):
        cc = convolve(cc, np.array([-rr[k], 1.0]))
    return cc


def vector_floater_hormann_rational_interpolation(tt, yy, degree):
    # len(tt) = num_pts
    # yy.shape = (num_pts, spatial_dimension)
    num_pts, spatial_dimension = yy.shape

    numerator = zeros_vector_polynomial(spatial_dimension)
    denominator = zeros_vector_polynomial(spatial_dimension)
    for ii in range(num_pts - degree):
        tt_i = tt[ii:ii+degree+1]
        print('yy.shape=', yy.shape)
        print('yy[ii:ii+degree+1, :].shape=', yy[ii:ii+degree+1, :].shape)
        print('len(tt_i)=', len(tt_i))
        yy_i = yy[ii:ii+degree+1, :].reshape((degree+1, spatial_dimension))
        p_i = vector_polynomial_interpolation(tt_i, yy_i)
        lambda_coeffs_1d = ((-1.)**ii) * polynomial_coeffs_from_roots(tt_i)
        lambda_coeffs = np.outer(np.ones(spatial_dimension), lambda_coeffs_1d)
        lambda_i = VectorPolynomial(lambda_coeffs)

        # numerator = numerator + lambda_i * p_i
        # denominator = denominator + lambda_i
        numerator = p_i
    denominator = ones_vector_polynomial(spatial_dimension)

    # rat = numerator / denominator
    rat = numerator
    return rat


