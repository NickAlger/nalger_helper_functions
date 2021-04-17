import numpy as np


def make_regular_grid(grid_min, grid_max, grid_shape, return_meshgrid=True, return_gridpoints=False):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/make_regular_grid.ipynb
    linear_grids = list(np.linspace(grid_min[k], grid_max[k], grid_shape[k]) for k in range(len(grid_min)))
    if return_meshgrid or return_gridpoints:
        meshgrids = np.meshgrid(*linear_grids, indexing='ij')

    if return_gridpoints:
        gridpoints = np.vstack([X.reshape(-1) for X in meshgrids]).T
        return linear_grids, meshgrids, gridpoints
    elif return_meshgrid:
        meshgrids = np.meshgrid(*linear_grids, indexing='ij')
        return linear_grids, meshgrids
    else:
        return linear_grids