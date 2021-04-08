import numpy as np
from scipy.interpolate import interpn
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class BoxFunction:
    box_min : np.array
    box_max : np.array
    data : np.array

    def __init__(me, min_pt, max_pt, data):
        me.box_min = np.asarray(np.array(min_pt)).reshape(-1)
        me.box_max = np.asarray(np.array(max_pt)).reshape(-1)
        me.data = data

        me.hh = (me.box_max - me.box_min) / (np.array(me.shape) - 1.)
        me.element_volume = np.prod(me.hh)

        me._lingrid = None
        me._meshgrid = None
        me._gridpoints = None

    @property
    def lingrid(me):
        if me._lingrid is None:
            me._lingrid = list(np.linspace(me.box_min[k], me.box_max[k], me.shape[k]) for k in range(me.ndim))
        return me._lingrid

    @property
    def _slightly_bigger_lingrid(me):
        return list(np.linspace(me.box_min[k]-1e-14, me.box_max[k]+1e-14, me.shape[k]) for k in range(me.ndim))

    @property
    def meshgrid(me):
        if me._meshgrid is None:
            me._meshgrid = np.meshgrid(*me._lingrid, indexing='ij')
        return me._meshgrid

    @property
    def gridpoints(me):
        if me._gridpoints is None:
            me._gridpoints = np.array([X.reshape(-1) for X in me.meshgrid]).T
        return me._gridpoints

    @property
    def shape(me):
        return me.data.shape

    @property
    def ndim(me):
        return me.data.ndim

    def __call__(me, pp, fill_value=0.0, method='linear'):
        return interpn(me._slightly_bigger_lingrid, me.data, pp, bounds_error=False, fill_value=fill_value, method=method)

    def __mul__(me, other):
        if isinstance(other, BoxFunction):
            if box_functions_are_conforming(me, other):
                new_min = np.max([me.box_min, other.box_min], axis=0)
                new_max = np.min([me.box_max, other.box_max], axis=0)
                F = me.restrict_to_another_box(new_min, new_max)
                G = other.restrict_to_another_box(new_min, new_max)
                new_data = F.data * G.data
                return BoxFunction(new_min, new_max, new_data)
            else:
                raise RuntimeError('BoxFunctions are not conforming')
        else:
            return BoxFunction(me.box_min, me.box_max, other * me.data)

    def __rmul__(me, other):
        return me.__mul__(other)

    def __add__(me, other):
        if isinstance(other, BoxFunction):
            if box_functions_are_conforming(me, other):
                new_min = np.min([me.box_min, other.box_min], axis=0)
                new_max = np.max([me.box_max, other.box_max], axis=0)
                F = me.restrict_to_another_box(new_min, new_max)
                G = other.restrict_to_another_box(new_min, new_max)
                new_data = F.data + G.data
                return BoxFunction(new_min, new_max, new_data)
            else:
                raise RuntimeError('BoxFunctions are not conforming')
        else:
            raise RuntimeError('currently BoxFunction can only be added to BoxFunction')

    def __radd__(me, other):
        return me.__add__(other)

    def __neg__(me):
        return BoxFunction(me.box_min, me.box_max, -me.data)

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

    def norm(me):
        return np.linalg.norm(me.data) * np.sqrt(me.element_volume)

    def restrict_to_another_box(me, new_min, new_max):
        if box_conforms_to_grid(new_min, new_max, me.box_min, me.hh):
            new_shape = tuple(np.round(new_max - new_min / me.hh).astype(int) + 1)
            new_F = BoxFunction(new_min, new_max, np.zeros(new_shape))
            new_F.data = me(new_F.gridpoints).reshape(new_shape)
        else:
            raise RuntimeError('other box not conforming with BoxFunction grid')
        return new_F

    def plot(me, title=None):
        plt.figure()
        X, Y = me.meshgrid
        plt.pcolor(X, Y, me.data)
        plt.colorbar()
        if title is not None:
            plt.title(title)


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

    if np.linalg.norm(F.hh - G.hh) > 1e-10:
        conforming = False
        print('BoxFunctions not conforming (different spacings h)')

    if not is_divisible_by(G.box_min - F.box_min, F.hh):
        conforming = False
        print('BoxFunctions not conforming (one grid is shifted relative to the other)')

    return conforming


def is_divisible_by(xx, yy, tol=1e-10):
    return np.linalg.norm(xx / yy - np.rint(xx / yy)) < tol