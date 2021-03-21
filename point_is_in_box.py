import numpy as np


def point_is_in_box(p, box_min, box_max):
    # https://github.com/NickAlger/helper_functions/blob/master/point_is_in_box.ipynb
    # p.shape = (N,d) for N points, OR p.shape = (d) for one point
    # box_min.shape = (d)
    # box_max.shape = (d)
    if len(p.shape) == 1:
        p = p.reshape((1,-1))
    box_min = box_min.reshape((1,-1))
    box_max = box_max.reshape((1,-1))
    return np.logical_and(np.all(box_min <= p, axis=1),
                          np.all(p <= box_max, axis=1)).reshape(-1)