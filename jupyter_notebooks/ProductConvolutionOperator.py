import numpy as np
from functools import cached_property

from nalger_helper_functions import BoxFunction, multilinear_interpolation_matrix, boxconv

# me.TT_grid2dof = list()
#         for k in range(me.num_pts):
#             conv_box_min = me.WW[k].min + me.FF[k].min
#             conv_box_max = me.WW[k].max + me.FF[k].max
#             conv_box_shape = None
#             T_g2d = multilinear_interpolation_matrix(me.pp, conv_box_min, conv_box_max, conv_box_shape)
#             me.TT_grid2dof.append(T_g2d)

class ProductConvolutionOperator:
    def __init__(me, WW, FF, input_inds, output_inds, TT_dof2grid, TT_grid2dof):
        # WW: weighting functions (a length-k list of BoxFunctions)
        # FF: convolution kernels (a length-k list of BoxFunctions)
        # pp: mesh dof points (num_pts x d array for num_pts dofs and spatial dimension d)
        #
        # input_inds: length-k list of arrays of indices for dofs that contribute to each convolution input
        #     u[input_inds[k]] = dof values from u that are relevant to WW[k]
        #
        # output_inds: length-k list of arrays of indices for dofs that are affected by each convolution output
        #     v[input_inds[k]] = dof values from v that are affected to boxconv(FF[k], WW[k])
        #
        # TT_dof2grid: list of sparse dof-to-grid transfer matrices.
        #     U = (TT_dof2grid[k] * u[input_inds[k]]).reshape(WW[k].shape)
        #     u is vector of function values at dof locations, u.shape=(num_pts,)
        #     U is array of function values on weighting function grid, U.shape=WW[k].shape
        #
        # TT_grid2dof: list of sparse grid-to-dof transfer matrices.
        #     q[output_inds[k]] = TT_dof2grid[k] * Q.reshape(-1)
        #     q is vector of function values at dof locations, q.shape=(num_pts,)
        #     Q is array of function values on convolution grid, Q.shape=boxconv(FF[k], WW[k]).shape
        me.WW = WW
        me.FF = FF
        me.input_inds = input_inds
        me.output_inds = output_inds
        me.TT_dof2grid = TT_dof2grid
        me.TT_grid2dof = TT_grid2dof

        me.num_pts = len(me.WW)
        me.d = WW[0].ndim
        me.shape = (me.num_pts, me.num_pts)

        me.FF_star = [F.flip().conj() for F in me.FF]

    def matvec(me, u):
        v = np.zeros(u.shape, dtype=u.dtype)
        for k in range(me.num_pts):
            Uk_array = (me.TT_dof2grid[k] * u[me.input_inds[k]]).reshape(me.WW[k].shape)
            WUk = BoxFunction(me.WW[k].min, me.WW[k].max, me.WW[k].array * Uk_array)
            Vk = boxconv(me.FF[k], WUk)
            v[me.output_inds[k]] += me.TT_grid2dof * Vk.reshape(-1)
        return v

    def rmatvec(me, v):
        pass

    def astype(me, dtype):
        WW_new = [W.astype(dtype) for W in me.WW]
        FF_new = [F.astype(dtype) for F in me.FF]
        TT_dof2grid_new = [T.astype(dtype) for T in me.TT_dof2grid]
        TT_grid2dof_new = [T.astype(dtype) for T in me.TT_grid2dof]
        return ProductConvolutionOperator(WW_new, FF_new,
                                          me.input_inds, me.output_inds,
                                          TT_dof2grid_new, TT_grid2dof_new)

