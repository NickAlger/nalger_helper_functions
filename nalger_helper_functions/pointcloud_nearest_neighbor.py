import numpy as np


def pointcloud_nearest_neighbor(pp, return_min_distances=False):
    # Using "dumb" algorithm. Probably slow for large numbers of points
    # https://github.com/NickAlger/helper_functions/blob/master/pointcloud_nearest_neighbor.ipynb
    N, d = pp.shape

    nearest_neighbor_inds = np.zeros(N, dtype=int)
    for k in range(N):
        p = pp[k,:].reshape((1, d))
        qq = np.vstack([pp[:k,:], [np.inf * np.ones(d)], pp[k+1:,:]]).reshape((N, d))
        nearest_neighbor_inds[k] = np.argmin(np.linalg.norm(qq - p, axis=1))

    if return_min_distances:
        nearest_neighbor_distances = np.linalg.norm(pp - pp[nearest_neighbor_inds,:], axis=1)
        return nearest_neighbor_inds, nearest_neighbor_distances
    else:
        return nearest_neighbor_inds