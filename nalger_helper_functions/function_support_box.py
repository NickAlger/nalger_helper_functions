import numpy as np


def function_support_box(function_values_uu, dof_coordinates_X, support_atol=1e-6, support_rtol=1e-3):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/blob/master/jupyter_notebooks/function_support_box.ipynb
    uu = function_values_uu
    X = dof_coordinates_X
    N, d = X.shape

    auu = np.abs(uu)
    u_nonzero_mask_rtol = (auu / np.max(auu) > support_rtol)
    u_nonzero_mask_atol = (auu > support_atol)
    u_nonzero_mask = np.logical_and(u_nonzero_mask_rtol, u_nonzero_mask_atol)

    nonzero_pts = X[u_nonzero_mask, :].reshape((-1, d))
    min_pt = np.min(nonzero_pts, axis=0)
    max_pt = np.max(nonzero_pts, axis=0)
    return min_pt, max_pt
