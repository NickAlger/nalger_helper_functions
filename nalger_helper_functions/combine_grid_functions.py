import numpy as np
from nalger_helper_functions import grid_interpolate, make_regular_grid, conforming_box


def combine_grid_functions(mins, maxes, AA, expand_box=True):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/combine_grid_functions.ipynb
    mins = np.array(mins)
    maxes = np.array(maxes)
    N = len(AA)
    d = len(AA[0].shape)

    mins = mins.reshape((-1,d))
    maxes = maxes.reshape((-1,d))
    anchor_point = mins[0,:]

    if expand_box:
        grid_hh = (maxes[0, :] - mins[0, :]) / (np.array(AA[0].shape) - 1.)
        B_min0 = np.min(mins, axis=0)
        B_max0 = np.max(maxes, axis=0)
        B_min, B_max, B_shape = conforming_box(B_min0, B_max0, anchor_point, grid_hh)
    else:
        B_min = mins[0,:]
        B_max = maxes[0,:]
        B_shape = AA[0].shape

    _, (X, Y) = make_regular_grid(B_min, B_max, B_shape)

    B_points = np.vstack([X.reshape(-1), Y.reshape(-1)]).T

    B = AA[0].copy()
    for k in range(1,N): # Earlier functions overwrite later functions
        B_nan_locations = np.isnan(B)
        if not np.any(B_nan_locations):
            break

        Ak_min = mins[k,:] - 1e-12
        Ak_max = maxes[k,:] + 1e-12
        Ak = AA[k]
        Ak_on_Bgrid = grid_interpolate(Ak_min, Ak_max, Ak, B_points, fill_value=np.nan).reshape(B_shape)

        B[B_nan_locations] = Ak_on_Bgrid[B_nan_locations]

    return B_min, B_max, B