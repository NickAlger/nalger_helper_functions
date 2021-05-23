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
        else:
            return VectorPolynomial(me.coeffs * other)

    def __radd__(me, other):
        return me + other

    def __neg__(me):
        return VectorPolynomial(-me.coeffs)

    def __sub__(me, other):
        return me + (-other)

    def __rsub__(me, other):
        return other + (-me)

    def derivative(me):
        dcoeffs = me.coeffs[:, 1:] * np.arange(1, me.k).reshape((1, -1))  # [pp[:,1], ..., pp[:,k]] * [[1,2,3,...,k]]
        return VectorPolynomial(dcoeffs)

    def arc_length(me, T, display_err=False, tangent=None):
        if tangent is None:
            tangent = me.derivative()
        ds = lambda t: np.linalg.norm(tangent(t))
        L, err = quad(ds, 0.0, T)
        if display_err:
            print('err=', err)
        return L

    def arc_time(me, L, display_soln=False, tangent=None, **kwargs):
        if me.k == 1: # curve is constant
            return np.inf

        tangent_length = np.linalg.norm(me.coeffs[:,1])

        if tangent is None:
            tangent = me.derivative()

        res = lambda T: L - me.arc_length(T, tangent=tangent)
        np.linalg.norm(me.coeffs[:,1])
        soln = root_scalar(res, x0=0.0, x1=0.5*L/tangent_length, **kwargs)
        if display_soln:
            print(soln)
        return soln.root


    def __eq__(me, other):
        return (me.coeffs == other.coeffs)

    def __hash__(me):
        return hash(me.coeffs)





def make_taylor_vector_polynomial(coeffs):
    return VectorPolynomial((1./factorial(np.arange(coeffs.shape[1]))).reshape((1,-1)) * coeffs)