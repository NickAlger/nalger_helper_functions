import numpy as np
from time import time


def build_dense_matrix_from_matvecs(apply_A, ncol_A, display=False, display_delay=1):
    e0 = np.zeros(ncol_A)
    e0[0] = 1.
    zeroth_col = apply_A(e0)
    nrow_A = len(zeroth_col)
    A_dense = np.zeros((nrow_A, ncol_A), dtype=zeroth_col.dtype)
    A_dense[:,0] = zeroth_col
    t_start = time()
    t_prev = t_start
    for k in range(1,ncol_A):
        ek = np.zeros(ncol_A)
        ek[k] = 1.
        A_dense[:,k] = apply_A(ek)
        if display:
            t_cur = time()
            if (t_cur - t_prev) > display_delay:
                time_elapsed = t_cur - t_start
                print(k, ' columns computed, ', time_elapsed, ' seconds elapsed.')
                t_prev = t_cur
    return A_dense

