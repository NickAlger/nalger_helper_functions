import numpy as np
import scipy.sparse as sps


def fit_sparse_matrix(apply_A, csr_indices, csr_indptr, shape, sampling_factor=1.0):
    max_nonzeros_per_row = np.max(csr_indptr[1:] - csr_indptr[:-1])
    m = int(max_nonzeros_per_row * sampling_factor)
    Omega = np.random.randn(shape[1], m)
    Y = np.vstack([apply_A(Omega[:,k]) for k in range(Omega.shape[1])]).T
    dtype = Y.dtype
    Omega = Omega.astype(dtype)
    values = list()
    for k in range(len(csr_indptr)-1):
        if csr_indptr[k+1] > csr_indptr[k]:
            cc = csr_indices[csr_indptr[k] : csr_indptr[k+1]]
            B = Y[k, :]
            M = Omega[cc, :]
            x = np.linalg.lstsq(M.T, B.T, rcond=None)[0] # min 0.5*||x^T*M - B||^2
            values.append(x)
    values = np.concatenate(values)
    A = sps.csr_matrix((values, csr_indices, csr_indptr), shape=shape)
    return A

