import numpy as np


def conforming_box(min_point, max_point, grid_zero_point, grid_h):
    # https://github.com/NickAlger/helper_functions/blob/master/conforming_rectangle.ipynb
    h = grid_h
    p = grid_zero_point

    min_point_conforming = np.floor((min_point - p) / h) * h + p
    max_point_conforming = np.ceil((max_point - p) / h) * h + p

    grid_shape = tuple(np.round((max_point_conforming - min_point_conforming) / h).astype(int) + 1)

    return min_point_conforming, max_point_conforming, grid_shape