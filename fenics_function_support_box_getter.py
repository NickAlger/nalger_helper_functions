import numpy as np


class FenicsFunctionSupportBoxGetter:
    # https://github.com/NickAlger/helper_functions/blob/master/fenics_function_support_box_getter.ipynb
    def __init__(me, function_space_V):
        me.V = function_space_V
        me.X = me.V.tabulate_dof_coordinates()
        me.N, me.d = me.X.shape

    def get_function_support_box(me, u, support_atol=1e-6, support_rtol=1e-3):
        au = np.abs(u.vector()[:])
        u_nonzero_mask_rtol = (au / np.max(au) > support_rtol)
        u_nonzero_mask_atol = (au > support_atol)
        u_nonzero_mask = np.logical_and(u_nonzero_mask_rtol, u_nonzero_mask_atol)

        nonzero_pts = me.X[u_nonzero_mask, :].reshape((-1, me.d))
        min_pt = np.min(nonzero_pts, axis=0)
        max_pt = np.max(nonzero_pts, axis=0)
        return min_pt, max_pt