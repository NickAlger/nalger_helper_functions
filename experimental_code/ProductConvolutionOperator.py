import numpy as np
from functools import cached_property

from nalger_helper_functions import *


def make_grid2dofs_transfer_matrix(Wk, Fk, pp):
    C = boxconv(Wk, Fk)
    inside_inds_k = np.argwhere(point_is_in_box(pp, C.min, C.max)).reshape(-1)
    T_grid2dof = multilinear_interpolation_matrix(pp[inside_inds_k, :], C.min, C.max, C.shape)
    return T_grid2dof, inside_inds_k

def make_dofs2grid_transfer_matrix(Wk, V):
    pass

class ProductConvolutionOperator:
    def __init__(me, WW, FF, shape, input_ind_groups, output_ind_groups, TT_dof2grid, TT_grid2dof):
        # WW: weighting functions (a length-k list of BoxFunctions)
        # FF: convolution kernels (a length-k list of BoxFunctions)
        # shape: shape of the operator. tuple. shape=(num_rows, num_cols)
        #
        # input_ind_groups: length-k list of arrays of indices for dofs that contribute to each convolution input
        #     u[input_ind_groups[k]] = dof values from u that are relevant to WW[k]
        #
        # output_ind_groups: length-k list of arrays of indices for dofs that are affected by each convolution output
        #     v[input_ind_groups[k]] = dof values from v that are affected to boxconv(FF[k], WW[k])
        #
        # TT_dof2grid: list of sparse dof-to-grid transfer matrices.
        #     U = (TT_dof2grid[k] * u[input_ind_groups[k]]).reshape(WW[k].shape)
        #     u is vector of function values at dof locations, u.shape=(num_pts,)
        #     U is array of function values on weighting function grid, U.shape=WW[k].shape
        #
        # TT_grid2dof: list of sparse grid-to-dof transfer matrices.
        #     q[output_ind_groups[k]] = TT_dof2grid[k] * Q.reshape(-1)
        #     q is vector of function values at dof locations, q.shape=(num_pts,)
        #     Q is array of function values on convolution grid, Q.shape=boxconv(FF[k], WW[k]).shape
        me.WW = WW
        me.FF = FF
        me.input_ind_groups = input_ind_groups
        me.output_ind_groups = output_ind_groups
        me.TT_dof2grid = TT_dof2grid
        me.TT_grid2dof = TT_grid2dof

        me.num_pts = len(me.WW)
        me.d = WW[0].ndim
        me.shape = shape

        me.FF_star = [F.flip().conj() for F in me.FF]

        me.dtype = dtype_max([W.dtype for W in me.WW] + [F.dtype for F in me.FF])

    def matvec(me, u):
        v = np.zeros(me.shape[0], dtype=dtype_max([me.dtype, u.dtype]))
        for k in range(me.num_pts):
            ii = me.input_ind_groups[k]
            oo = me.output_ind_groups[k]
            Ti = me.TT_dof2grid[k]
            To = me.TT_grid2dof
            Wk = me.WW[k]
            Fk = me.FF[k]

            Uk_array = (Ti * u[ii]).reshape(Wk.shape)
            WUk = BoxFunction(Wk.min, Wk.max, Wk.array * Uk_array)
            Vk = boxconv(Fk, WUk)
            v[oo] += To * Vk.reshape(-1)
        return v

    def rmatvec(me, v):
        u = np.zeros(me.shape[1], dtype=dtype_max([me.dtype, v.dtype]))
        for k in range(me.num_pts):
            ii = me.input_ind_groups[k]
            oo = me.output_ind_groups[k]
            Ti = me.TT_dof2grid[k]
            To = me.TT_grid2dof
            Wk = me.WW[k]
            Fk_star = me.FF_star[k]

            Vk = To.T * v[oo]
            Sk = boxconv(Fk_star, Vk)
            Uk_big = Sk * Wk
            Uk = Uk_big.restrict_to_new_box(Wk.min, Wk.max)
            u[ii] += Ti.T * Uk.reshape(-1)
        return u

    def astype(me, dtype):
        WW_new = [W.astype(dtype) for W in me.WW]
        FF_new = [F.astype(dtype) for F in me.FF]
        TT_dof2grid_new = [T.astype(dtype) for T in me.TT_dof2grid]
        TT_grid2dof_new = [T.astype(dtype) for T in me.TT_grid2dof]
        return ProductConvolutionOperator(WW_new, FF_new,
                                          me.input_ind_groups, me.output_ind_groups,
                                          TT_dof2grid_new, TT_grid2dof_new)

    @property
    def real(me):
        WW_new = [W.real for W in me.WW]
        FF_new = [F.real for F in me.FF]
        TT_dof2grid_new = [T.real for T in me.TT_dof2grid]
        TT_grid2dof_new = [T.real for T in me.TT_grid2dof]
        return ProductConvolutionOperator(WW_new, FF_new,
                                          me.input_ind_groups, me.output_ind_groups,
                                          TT_dof2grid_new, TT_grid2dof_new)

    @property
    def imag(me):
        FF_new = [F.imag for F in me.FF]
        return ProductConvolutionOperator(me.WW, FF_new,
                                          me.input_ind_groups, me.output_ind_groups,
                                          me.TT_dof2grid, me.TT_grid2dof)


