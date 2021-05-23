import numpy as np
from scipy.signal import fftconvolve
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
        denominator = VectorPolynomial(np.ones((me.N, 1)))
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