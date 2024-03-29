import numpy as np
from scipy.interpolate import interpn
from scipy.signal import convolve
from scipy import fft
import matplotlib.pyplot as plt

from nalger_helper_functions import dtype_max, point_is_in_ellipsoid


class BoxFunction:
    def __init__(me, box_min, box_max, array):
        me.min = np.asarray(np.array(box_min)).reshape(-1)
        me.max = np.asarray(np.array(box_max)).reshape(-1)
        me.array = array

        me.width = me.max - me.min
        me.h = me.width / (np.array(me.shape) - 1.)
        me.dV = np.prod(me.h)

        me.zeropoint_index = me.nearest_gridpoint_index(np.zeros(me.ndim))

        me._lingrid = None
        me._slightly_bigger_lingrid = None
        me._meshgrid = None
        me._gridpoints = None

    @property
    def lingrid(me):
        if me._lingrid is None:
            me._lingrid = list(np.linspace(me.min[k], me.max[k], me.shape[k]) for k in range(me.ndim))
        return me._lingrid

    @property
    def slightly_bigger_lingrid(me):
        if me._slightly_bigger_lingrid is None:
            me._slightly_bigger_lingrid = list(np.linspace(me.min[k] - 1e-15, me.max[k] + 1e-15, me.shape[k]) for k in range(me.ndim))
        return me._slightly_bigger_lingrid

    @property
    def meshgrid(me):
        if me._meshgrid is None:
            me._meshgrid = np.meshgrid(*me.lingrid, indexing='ij')
        return me._meshgrid

    @property
    def gridpoints(me):
        if me._gridpoints is None:
            me._gridpoints = np.array([X.reshape(-1) for X in me.meshgrid]).T
        return me._gridpoints

    @property
    def shape(me):
        return me.array.shape

    @property
    def ndim(me):
        return me.array.ndim

    def __call__(me, pp, fill_value=0.0, method='linear'):
        return interpn(me.slightly_bigger_lingrid, me.array, pp, bounds_error=False, fill_value=fill_value, method=method)

    def __mul__(me, other):
        if isinstance(other, BoxFunction):
            if not box_functions_are_conforming(me, other):
                raise RuntimeError('BoxFunctions are not conforming')

            new_min = np.max([me.min, other.min], axis=0)
            new_max = np.min([me.max, other.max], axis=0)
            F = me.restrict_to_new_box(new_min, new_max)
            G = other.restrict_to_new_box(new_min, new_max)
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
            F = me.restrict_to_new_box(new_min, new_max)
            G = other.restrict_to_new_box(new_min, new_max)
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

    def restrict_to_new_box(me, new_min, new_max, check_error=False):
        if not box_conforms_to_grid(new_min, new_max, me.min, me.h):
            raise RuntimeError('other box not conforming with BoxFunction grid')

        new_shape = tuple(np.round((new_max - new_min) / me.h).astype(int) + 1)
        new_F = BoxFunction(new_min, new_max, np.zeros(new_shape, dtype=me.dtype))

        intersection_min = np.max([me.min, new_min], axis=0)
        intersection_max = np.min([me.max, new_max], axis=0)
        if np.all(intersection_max >= intersection_min):
            min_ind_old = me.nearest_gridpoint_index(intersection_min)
            max_ind_old = me.nearest_gridpoint_index(intersection_max)
            intersection_slices_old = tuple([slice(min_ind_old[k], max_ind_old[k]+1)
                                             for k in range(me.ndim)])

            min_ind_new = new_F.nearest_gridpoint_index(intersection_min)
            max_ind_new = new_F.nearest_gridpoint_index(intersection_max)
            intersection_slices_new = tuple([slice(min_ind_new[k], max_ind_new[k]+1)
                                             for k in range(me.ndim)])

            new_F.array[intersection_slices_new] = me.array[intersection_slices_old]

        if check_error:
            new_F_array2 = me(new_F.gridpoints).reshape(new_shape)
            err_restrict_to_new_box = np.linalg.norm(new_F.array - new_F_array2)
            print('err_restrict_to_new_box=', err_restrict_to_new_box)

        return new_F

    def translate(me, p):
        return BoxFunction(me.min + p, me.max + p, me.array)

    def nearest_gridpoint_index(me, pp):
        return np.round((pp - me.min) / me.h).astype(int)

    def plot(me, title=None, figsize=None):
        X, Y = me.meshgrid

        if me.dtype == 'complex128':
            if figsize is None:
                figsize = (12, 4)
            plt.figure(figsize=figsize)

            plt.subplot(121)
            plt.pcolor(X, Y, me.array.real)
            plt.colorbar()
            if title is not None:
                plt.title(title + ' real')

            plt.subplot(122)
            plt.pcolor(X, Y, me.array.imag)
            plt.colorbar()
            if title is not None:
                plt.title(title + ' imag')
        else:
            if figsize is None:
                figsize = (6, 4)
            plt.figure(figsize=figsize)

            plt.pcolor(X, Y, me.array)
            plt.colorbar()
            if title is not None:
                plt.title(title)


def boxconv(F, G, method='auto'):
    # if not box_functions_are_conforming(F, G):
    #     raise RuntimeError('BoxFunctions are not conforming')
    if np.linalg.norm(F.h - G.h) > 1e-10:
        raise RuntimeError('BoxFunctions have different spacings h')

    dtype = dtype_max([F, G])
    F_star_G_data = convolve(F.array.astype(dtype), G.array.astype(dtype), mode='full', method=method) * F.dV
    return BoxFunction(F.min + G.min, F.max + G.max, F_star_G_data)


def boxinner(F, G):
    return boxintegrate(F * G.conj())


def boxnorm(F):
    return np.sqrt(np.abs(boxinner(F,F)))


def boxintegrate(F):
    return np.sum(F.array) * F.dV


def ellipsoid_characteristic_function(box_min, box_max, grid_shape, mu, Sigma, tau):
    E = BoxFunction(box_min, box_max, np.zeros(grid_shape))
    E.array = point_is_in_ellipsoid(E.gridpoints, mu, Sigma, tau).reshape(grid_shape).astype(float)
    return E


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


def convolution_square_root(PSI, pre_expansion=0, post_contraction=0,
                            positive_real_branch_cut=np.pi, negative_real_branch_cut=0.0):
    # Input PSI and output Z are BoxFunction convolution kernels
    # Z is the convolution kernel for the square root of the operator that does convolution with PSI
    initial_width = PSI.width
    s = pre_expansion
    PSI = PSI.restrict_to_new_box(PSI.min - s*PSI.width, PSI.max + s*PSI.width)  # expand PSI box by zero

    PSI_shifted_array = np.roll(PSI.array, -PSI.zeropoint_index, axis=np.arange(PSI.ndim))

    fft_PSI = fft.fftn(PSI_shifted_array)
    sqrt_fft_PSI = np.zeros(fft_PSI.shape, dtype=complex)
    sqrt_fft_PSI[fft_PSI.real >= 0] = square_root(fft_PSI[fft_PSI.real >= 0], positive_real_branch_cut)
    sqrt_fft_PSI[fft_PSI.real < 0] = square_root(fft_PSI[fft_PSI.real < 0], negative_real_branch_cut)

    Z_shifted_array = fft.ifftn(sqrt_fft_PSI)

    Z_array = np.roll(Z_shifted_array, PSI.zeropoint_index, axis=np.arange(PSI.ndim)) / np.sqrt(PSI.dV)

    Z = BoxFunction(PSI.min, PSI.max, Z_array)

    t = post_contraction
    Z = Z.restrict_to_new_box(Z.min + t*initial_width, Z.max - t*initial_width)  # contract Z box
    return Z


def square_root(z, branch_cut_theta):
    "Square root with different branch cut defined by theta parameter."
    # https://flothesof.github.io/branch-cuts-with-square-roots.html
    argument = np.angle(z)  # between -pi and +pi
    modulus = np.abs(z)
    argument = np.mod(argument + branch_cut_theta, 2 * np.pi) - branch_cut_theta
    return np.sqrt(modulus) * np.exp(1j * argument / 2)