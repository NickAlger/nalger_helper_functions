import numpy as np
import scipy.sparse as sps
from scipy.interpolate import interpn

from nalger_helper_functions import fit_sparse_matrix


def multilinear_interpolation_matrix(pp, box_min, box_max, box_shape):
    pp = np.array(pp)
    if len(pp.shape) == 1:
        pp = pp.reshape((1,-1))
    box_min = np.array(box_min).reshape(-1)
    box_max = np.array(box_max).reshape(-1)

    num_pts, d = pp.shape

    h = (box_max - box_min) / (np.array(box_shape) - 1.)
    lower_inds = np.floor((pp - box_min) / h).astype(int)
    corner_multi_inds = np.array([lower_inds + x.reshape((1,-1)) for x in unit_box_corners(d)]) # 2^d x num_pts x d

    corner_inds = np.zeros((num_pts, 2**d), dtype=int)
    for k in range(corner_multi_inds.shape[0]):
        corner_multi_indices = tuple([corner_multi_inds[k,:,i].reshape(-1) for i in range(d)])
        corner_inds[:,k] = np.ravel_multi_index(corner_multi_indices, box_shape, mode='clip')

    csr_indices = corner_inds.reshape(-1)
    csr_indptr = (2**d) * np.arange(num_pts+1)

    all_lingrids = tuple([np.linspace(box_min[i], box_max[i], box_shape[i]) for i in range(d)])
    apply_A = lambda x: interpn(all_lingrids, x.reshape(box_shape), pp,
                                bounds_error=False, fill_value=0.0, method='linear')

    A_shape = (num_pts, np.prod(box_shape))
    A = fit_sparse_matrix(apply_A, csr_indices, csr_indptr, A_shape, sampling_factor=1.0)
    return A


def unit_box_corners(d):
    shape = tuple(2*np.ones(d, dtype=int))
    corners = list()
    for k in range(2**d):
        corners.append(np.array(np.unravel_index(k, shape)))
    return corners
