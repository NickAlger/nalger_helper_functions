import numpy as np
import typing as typ


def rsvd_double_pass(
        A_shape: typ.Tuple[int, int],
        A_matvecs_func: typ.Callable[[np.ndarray], np.ndarray],  # X -> A X, A has shape (N,M), X has shape (M, k1)
        A_rmatvecs_func: typ.Callable[[np.ndarray], np.ndarray], # Z -> Z A, A has shape (N,M), Z has shape (k2, N)
        rank: int,
        oversampling_parameter: int,
) -> typ.Tuple[
    np.ndarray, # U, shape=(N,r)
    np.ndarray, # ss, shape=(r,)
    np.ndarray, # Vt, shape=(r,M)
]:
    '''See Algorithms 4.1 and 5.1 in:
    Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
    "Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix decompositions."
    SIAM review 53.2 (2011): 217-288.
    '''
    N, M = A_shape
    Omega = np.random.randn(M, rank + oversampling_parameter)
    Y = A_matvecs_func(Omega)
    Q, R = np.linalg.qr(Y)
    B = A_rmatvecs_func(Q.T)
    U0, ss0, Vt0 = np.linalg.svd(B, full_matrices=False)
    U = Q @ U0[:, :rank]
    ss = ss0[:rank]
    Vt = Vt0[:rank,:]
    return U, ss, Vt
