import numpy as np
from scipy.spatial import cKDTree

from nalger_helper_functions import point_is_in_box


def shortest_distance_between_points_in_box(box_min, box_max, pp):
    box_mask = point_is_in_box(pp, box_min, box_max)
    points_in_box = pp[box_mask, :]

    T = cKDTree(points_in_box)
    dd, _ = T.query(points_in_box, k=2)
    h = np.min(dd[:,1])
    return h