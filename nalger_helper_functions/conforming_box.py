import numpy as np


def conforming_box(min_point, max_point, grid_zero_point, grid_hh, rtol=1e-12):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/blob/master/jupyter_notebooks/conforming_box.ipynb
    h = grid_hh
    p = grid_zero_point

    min_grid_coords = (min_point - p) / h
    max_grid_coords = (max_point - p) / h

    bad_inds_min = np.abs(min_grid_coords - np.round(min_grid_coords)) > rtol * np.abs(min_grid_coords)
    bad_inds_max = np.abs(max_grid_coords - np.round(max_grid_coords)) > rtol * np.abs(max_grid_coords)

    min_point_conforming = min_point.copy()
    max_point_conforming = max_point.copy()

    min_point_conforming[bad_inds_min] = (np.floor(min_grid_coords) * h + p)[bad_inds_min]
    max_point_conforming[bad_inds_max] = (np.ceil(max_grid_coords) * h + p)[bad_inds_max]

    grid_shape = tuple(np.round((max_point_conforming - min_point_conforming) / h).astype(int) + 1)

    return min_point_conforming, max_point_conforming, grid_shape