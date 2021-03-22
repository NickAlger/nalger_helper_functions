import numpy as np
import numba as nb  # <--- Remove if you don't have numba


@nb.njit  # <--- Remove if you don't have numba
def intersection_of_many_sets(SS):
    S_intersection = set(SS[0])
    for k in range(1, len(SS)):
        #         S_intersection = np.intersect1d(S_intersection, SS[k]) # np.intersect1d not supported in numba
        S_intersection = S_intersection.intersection(set(SS[k]))
    return list(S_intersection)


@nb.njit  # <--- Remove if you don't have numba
def get_inds_in_boxes(qq, box_mins, box_maxes):
    # See: https://github.com/NickAlger/helper_functions/blob/master/points_in_boxes.ipynb
    N, d = box_mins.shape

    qq_cartesian_sort_inds = [np.argsort(qq[:, k]) for k in range(d)]
    qq_cartesian_sorts = [qq[qq_cartesian_sort_inds[k], k] for k in range(d)]

    ll = np.empty((N, d))
    rr = np.empty((N, d))
    for k in range(d):
        ll[:, k] = np.searchsorted(qq_cartesian_sorts[k], box_mins[:, k])
        rr[:, k] = np.searchsorted(qq_cartesian_sorts[k], box_maxes[:, k])

    all_box_inds = list()
    for i in range(N):

        candidate_inds_by_dim = list()
        for k in range(d):
            candidate_inds_by_dim.append(qq_cartesian_sort_inds[k][ll[i, k]: rr[i, k]])

        inds_in_ith_box = intersection_of_many_sets(candidate_inds_by_dim)
        all_box_inds.append(inds_in_ith_box)

    return all_box_inds