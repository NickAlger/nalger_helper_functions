import numpy as np
from scipy.interpolate import interpn
from scipy.signal import convolve
from dataclasses import dataclass
from functools import cached_property
import matplotlib.pyplot as plt


@dataclass
class BoxFunction:
    min : np.array
    max : np.array
    array : np.array

    def __init__(me, box_min, box_max, array):
        me.min = np.asarray(np.array(box_min)).reshape(-1)
        me.max = np.asarray(np.array(box_max)).reshape(-1)
        me.array = array

        me.width = me.max - me.min
        me.h = me.width / (np.array(me.shape) - 1.)
        me.dV = np.prod(me.h)

        me.zeropoint_index = me.nearest_gridpoint_index(np.zeros(me.ndim))

    @cached_property
    def lingrid(me):
        return list(np.linspace(me.min[k], me.max[k], me.shape[k]) for k in range(me.ndim))

    @cached_property
    def _slightly_bigger_lingrid(me):
        return list(np.linspace(me.min[k] - 1e-15, me.max[k] + 1e-15, me.shape[k]) for k in range(me.ndim))

    @cached_property
    def meshgrid(me):
        return np.meshgrid(*me.lingrid, indexing='ij')

    @cached_property
    def gridpoints(me):
        return np.array([X.reshape(-1) for X in me.meshgrid]).T

    @property
    def shape(me):
        return me.array.shape

    @property
    def ndim(me):
        return me.array.ndim

    def __call__(me, pp, fill_value=0.0, method='linear'):
        return interpn(me._slightly_bigger_lingrid, me.array, pp, bounds_error=False, fill_value=fill_value, method=method)

    def __mul__(me, other):
        if isinstance(other, BoxFunction):
            if not box_functions_are_conforming(me, other):
                raise RuntimeError('BoxFunctions are not conforming')

            new_min = np.max([me.min, other.min], axis=0)
            new_max = np.min([me.max, other.max], axis=0)
            F = me.restrict_to_box(new_min, new_max)
            G = other.restrict_to_box(new_min, new_max)
            new_data = F.array * G.array
            return BoxFunction(new_min, new_max, new_data)
        else:
            return BoxFunction(me.min, me.max, other * me.array)

    def __rmul__(me, other):
        return me.__mul__(other)

    def __truediv__(me, scalar):
        if isinstance(scalar, BoxFunction):
            raise RuntimeError('BoxFunction can only be divided by scalar')

        return BoxFunction(me.min, me.max, me.array / scalar)

    def __add__(me, other):
        if isinstance(other, BoxFunction):
            if not box_functions_are_conforming(me, other):
                raise RuntimeError('BoxFunctions are not conforming')

            new_min = np.min([me.min, other.min], axis=0)
            new_max = np.max([me.max, other.max], axis=0)
            F = me.restrict_to_box(new_min, new_max)
            G = other.restrict_to_box(new_min, new_max)
            new_data = F.array + G.array
            return BoxFunction(new_min, new_max, new_data)
        else:
            raise RuntimeError('currently BoxFunction can only be added to BoxFunction')

    def __radd__(me, other):
        return me.__add__(other)

    def __neg__(me):
        return BoxFunction(me.min, me.max, -me.array)

    def __sub__(me, other):
        if isinstance(other, BoxFunction):
            return me.__add__(other.__neg__())
        else:
            raise RuntimeError('currently BoxFunction can only be subtracted from BoxFunction')

    def __rsub__(me, other):
        if isinstance(other, BoxFunction):
            return other.__add__(me.__neg__())
        else:
            raise RuntimeError('currently BoxFunction can only be subtracted from BoxFunction')

    @property
    def real(me):
        return BoxFunction(me.min, me.max, me.array.real)

    @property
    def imag(me):
        return BoxFunction(me.min, me.max, me.array.imag)

    def angle(me):
        return BoxFunction(me.min, me.max, np.angle(me.array))

    def abs(me):
        return BoxFunction(me.min, me.max, np.abs(me.array))

    def conj(me):
        return BoxFunction(me.min, me.max, me.array.conj())

    @property
    def dtype(me):
        return me.array.dtype

    def astype(me, *args, **kwargs):
        return BoxFunction(me.min, me.max, me.array.astype(*args, **kwargs))

    def copy(me):
        return BoxFunction(me.min, me.max, me.array.copy())

    def norm(me):
        return boxnorm(me)

    def inner(me, other):
        return boxinner(me, other)

    def flip(me):
        return BoxFunction(-me.max, -me.min, flip_array(me.array))

    def restrict_to_box(me, new_min, new_max):
        if not box_conforms_to_grid(new_min, new_max, me.min, me.h):
            raise RuntimeError('other box not conforming with BoxFunction grid')

        new_shape = tuple(np.round((new_max - new_min) / me.h).astype(int) + 1)
        new_F = BoxFunction(new_min, new_max, np.zeros(new_shape))
        new_F.array = me(new_F.gridpoints).reshape(new_shape)
        return new_F

    def translate(me, p):
        return BoxFunction(me.min + p, me.max + p, me.array)

    def nearest_gridpoint_index(me, pp):
        return np.round(pp - me.min / me.h).astype(int)

    def plot(me, title=None):
        plt.figure()
        X, Y = me.meshgrid
        plt.pcolor(X, Y, me.array)
        plt.colorbar()
        if title is not None:
            plt.title(title)


def boxconv(F, G, method='auto'):
    if not box_functions_are_conforming(F, G):
        raise RuntimeError('BoxFunctions are not conforming')

    F_star_G_data = convolve(F.array, G.array, mode='full', method=method) * F.dV
    return BoxFunction(F.min + G.min, F.max + G.max, F_star_G_data)


def boxinner(F, G):
    return np.sum((F * G.conj()).array * F.dV)


def boxnorm(F):
    return np.sqrt(np.abs(boxinner(F,F)))


def box_conforms_to_grid(box_min, box_max, anchor_point, hh):
    conforming = True

    if not is_divisible_by(box_min - anchor_point, hh):
        conforming = False
        print('box_min is not on grid')

    if not is_divisible_by(box_max - box_min, hh):
        conforming = False
        print('Box widths do not conform to grid spacings hh')

    return conforming


def box_functions_are_conforming(F, G):
    conforming = True

    if np.linalg.norm(F.h - G.h) > 1e-10:
        conforming = False
        print('BoxFunctions not conforming (different spacings h)')

    if not is_divisible_by(G.min - F.min, F.h):
        conforming = False
        print('BoxFunctions not conforming (one grid is shifted relative to the other)')

    return conforming


def is_divisible_by(xx, yy, tol=1e-10):
    return np.linalg.norm(xx / yy - np.rint(xx / yy)) < tol


def flip_array(X):
    return X[tuple([slice(None, None, -1) for _ in range(X.ndim)])]