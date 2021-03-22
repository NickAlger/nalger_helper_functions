import numpy as np
from scipy.interpolate import interpn


def grid_interpolate(grid_min, grid_max, F_grid, target_points_pp, fill_value=0.0, method='linear'):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/grid_interpolate.ipynb
    grid_shape = F_grid.shape
    d = len(grid_shape)
    all_xx = tuple([np.linspace(grid_min[i], grid_max[i], grid_shape[i]) for i in range(d)])
    return interpn(all_xx, F_grid, target_points_pp, bounds_error=False, fill_value=fill_value, method=method)