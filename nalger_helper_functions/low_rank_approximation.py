import numpy as np
import typing as typ

def low_rank_approximation(
        A: np.ndarray, # shape=(N, M)
        rtol: float = 1e-5,
        max_rank: int = 100,
) -> typ.Tuple[
    np.ndarray, # left_factor, shape=(N, rank)
    np.ndarray, # right_factor, shape=(rank, M)
]:
    '''||A - left_factor @ right_factor|| <= rtol * ||A|| in induced norm'''
    U, ss, Vt = np.linalg.svd(A, full_matrices=False)
    rank0 = np.sum(ss > rtol * ss[0])
    if rank0 == 0:
        left_factor = np.zeros((A.shape[0],1))
        right_factor = np.zeros((1, A.shape[1]))
    else:
        rank = np.minimum(rank0, max_rank)
        left_factor = U[:, :rank]
        right_factor = ss[:rank].reshape((-1,1)) * Vt[:rank, :]
    return left_factor, right_factor


# #### Example:
# N = 1000
# M = 700
# U, _, _ = np.linalg.svd(np.random.randn(N,M), full_matrices=False)
# ss = np.logspace(-12, 3, M)
# _, _, Vt = np.linalg.svd(np.random.randn(N,M), full_matrices=False)
# A = U @ np.diag(ss) @ Vt
#
# for rtol in np.logspace(-6, 0, 20)[::-1]:
#     left_factor, right_factor = low_rank_approximation(A, rtol=rtol, max_rank=200)
#     rank = left_factor.shape[1]
#     A2 = left_factor @ right_factor
#     err = np.linalg.norm(A - A2) / np.linalg.norm(A)
#     print('rtol=', rtol, ', rank=', rank, ', err=', err)