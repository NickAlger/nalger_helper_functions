import numpy as np
from scipy.interpolate import interpn


class GridInterpolator:
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/grid_interpolator.ipynb
    def __init__(me, grid_min, grid_max, grid_shape):
        me.grid_min = grid_min
        me.grid_max = grid_max
        me.grid_shape = grid_shape
        me.d = len(me.grid_shape)

        me.all_xx = tuple([np.linspace(me.grid_min[i],
                                       me.grid_max[i],
                                       me.grid_shape[i]) for i in range(me.d)])

    def interpolate(me, F_grid, points_array_pp, fill_value=0.0, method='linear'):
        return interpn(me.all_xx, F_grid, points_array_pp,
                       bounds_error=False, fill_value=fill_value, method=method)
