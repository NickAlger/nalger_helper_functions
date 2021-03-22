import numpy as np
from tqdm.auto import tqdm


def build_dense_matrix_from_matvecs(apply_A, ncol_A):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/blob/master/jupyter_notebooks/build_dense_matrix_from_matvecs.ipynb
    e0 = np.zeros(ncol_A)
    e0[0] = 1.
    zeroth_col = apply_A(e0)
    nrow_A = len(zeroth_col)
    A_dense = np.zeros((nrow_A, ncol_A), dtype=zeroth_col.dtype)
    A_dense[:,0] = zeroth_col

    for k in tqdm(range(1,ncol_A)):
        ek = np.zeros(ncol_A)
        ek[k] = 1.
        A_dense[:,k] = apply_A(ek)

    return A_dense