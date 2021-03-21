import numpy as np


def make_regular_grid(grid_min, grid_max, grid_shape, return_meshgrid=True):
    # https://github.com/NickAlger/helper_functions/blob/master/make_regular_grid.ipynb
    linear_grids = list(np.linspace(grid_min[k], grid_max[k], grid_shape[k]) for k in range(grid_min.shape[0]))
    if return_meshgrid:
        meshgrids = np.meshgrid(*linear_grids, indexing='ij')
        return linear_grids, meshgrids
    else:
        return linear_grids