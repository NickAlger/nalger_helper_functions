import numpy as np
import typing as typ

import typing as typ


def aca_full(
        A: np.ndarray,
        max_rank: int = 100,
        min_rank: int = 0,
        rtol: float = 1e-2,
        display=False
) -> typ.Tuple[typ.List[int],  # row_inds=[i1, i2, ..., ik]
               typ.List[int],  # col_inds=[j1, j2, ...m jk]
               typ.List[np.ndarray],  # uu = [u1, u2, ..., uk]
               typ.List[np.ndarray]  # vvt= [v1, v2, ..., vk]
]:
    '''Constructs low rank approximation of matrix A by sampling rows and columns.
    A =approx= u1 @ v1.T + u2 @ v2.T + ... + uk @ vk.T

    See Algorithm 3 (ACA with full pivoting) on page 15 in:
    Jonas Ballani and Daniel Kressner.
    "Matrices with hierarchical low-rank structures".
    https://sma.epfl.ch/~anchpcommon/publications/cime.pdf

    Looks at all matrix entries.
    Slower but more robust than aca_partial

    Example:
    In:
        B = np.zeros((500, 375))
        for ii in range(B.shape[0]):
            for jj in range(B.shape[1]):
                B[ii, jj] = 1.0 / (1.0 + ii + jj) # Hilbert tensor

        row_inds, col_inds, uu, ww = aca_full(B, rtol=1e-2, display=True)

        rank = len(row_inds)
        U = np.array(uu).T
        Wt = np.array(ww)
        B_aca = U @ Wt
        err_aca = np.linalg.norm(B_aca - B) / np.linalg.norm(B)

        U_svd, ss_svd, Vt_svd = np.linalg.svd(B, 0)
        B_svd = U_svd[:,:rank] @ np.diag(ss_svd[:rank]) @ Vt_svd[:rank, :]
        err_svd = np.linalg.norm(B_svd - B) / np.linalg.norm(B)
        print('rank=', rank, ', err_aca=', err_aca, ', err_svd=', err_svd)
    Out:
        k= 0 , relerr= 0.6711123226594852
        k= 1 , relerr= 0.47056887361563193
        k= 2 , relerr= 0.2538276424047767
        k= 3 , relerr= 0.24149704757381296
        k= 4 , relerr= 0.05398221821910914
        k= 5 , relerr= 0.003807438101234531
        rank= 6 , err_aca= 0.0038074381012345314 , err_svd= 0.0009653646620907224
    '''
    max_rank = np.min([np.min(A.shape), max_rank])
    R = A.copy()
    row_inds = []
    col_inds = []
    uu = []
    vv = []
    norm_A = np.linalg.norm(A)
    if norm_A == 0.0:
        if display:
            print('A is zero')
        return row_inds, col_inds, uu, vv

    for k in range(max_rank):
        ii, jj = np.unravel_index(np.argmax(np.abs(R), axis=None), R.shape)
        row_inds.append(ii)
        col_inds.append(jj)
        delta = R[ii, jj]
        u = R[:, jj].copy().reshape(-1)
        vt = R[ii, :].copy().reshape(-1) / delta
        uu.append(u)
        vv.append(vt)
        R -= np.outer(u, vt)
        norm_R = np.linalg.norm(R)
        relerr = norm_R / norm_A
        if display:
            print('k=', k, ', relerr=', relerr)
        if norm_R <= rtol * norm_A and len(row_inds) >= min_rank:
            break
    return row_inds, col_inds, uu, vv


def aca_partial(
        A_get_row: typ.Callable[[int], np.ndarray],
        A_get_col: typ.Callable[[int], np.ndarray],
        A_shape: typ.Tuple[int, int],
        max_rank: int = 100,
        rtol: float = 1e-2,
        display=False,
        error_estimator_rolling_window_size: int = 5
) -> typ.Tuple[typ.List[int],  # row_inds=[i1, i2, ..., ik]
               typ.List[int],  # col_inds=[j1, j2, ...m jk]
               typ.List[np.ndarray],  # uu = [u1, u2, ..., uk]
               typ.List[np.ndarray]  # vvt= [v1, v2, ..., vk]
]:
    '''Constructs low rank approximation of matrix A by sampling rows and columns.
    A =approx= u1 @ v1.T + u2 @ v2.T + ... + uk @ vk.T

    See Algorithm 4 (ACA with partial pivoting) on page 16 in:
    Jonas Ballani and Daniel Kressner.
    "Matrices with hierarchical low-rank structures".
    https://sma.epfl.ch/~anchpcommon/publications/cime.pdf

    Only looks at matrix entries in rows and columns used in the approximation.
    Faster but less robust than aca_full

    Example:
    In:
        B = np.zeros((500, 375))
        for ii in range(B.shape[0]):
            for jj in range(B.shape[1]):
                B[ii, jj] = 1.0 / (1.0 + ii + jj) # Hilbert tensor

        B_get_row = lambda ii: B[ii,:]
        B_get_col = lambda jj: B[:,jj]

        row_inds, col_inds, uu, ww = aca_partial(B_get_row, B_get_col, B.shape, rtol=1e-2, display=True)

        rank = len(row_inds)
        U = np.array(uu).T
        Wt = np.array(ww)
        B_aca = U @ Wt
        err_aca = np.linalg.norm(B_aca - B) / np.linalg.norm(B)

        U_svd, ss_svd, Vt_svd = np.linalg.svd(B, 0)
        B_svd = U_svd[:,:rank] @ np.diag(ss_svd[:rank]) @ Vt_svd[:rank, :]
        err_svd = np.linalg.norm(B_svd - B) / np.linalg.norm(B)
        print('rank=', rank, ', err_aca=', err_aca, ', err_svd=', err_svd)
    Out:
        k= 0 , relerr_estimate= 1.0000000000000002 , num_successes= 0
        k= 1 , relerr_estimate= 0.9414152902898599 , num_successes= 0
        k= 2 , relerr_estimate= 0.038759420958288666 , num_successes= 0
        k= 3 , relerr_estimate= 0.02903746410600461 , num_successes= 0
        k= 4 , relerr_estimate= 0.04551107924027933 , num_successes= 0
        k= 5 , relerr_estimate= 0.0026704977816689197 , num_successes= 0
        k= 6 , relerr_estimate= 0.002351820583877269 , num_successes= 1
        k= 7 , relerr_estimate= 0.0012760090490301702 , num_successes= 2
        k= 8 , relerr_estimate= 9.351347289873483e-05 , num_successes= 3
        k= 9 , relerr_estimate= 3.835674291368529e-05 , num_successes= 4
        rtol= 0.01  achieved.
        rank= 10 , err_aca= 2.5737653108161734e-05 , err_svd= 3.2868723166024736e-06
    '''
    rtol_squared = rtol * rtol
    max_rank = np.min([np.min(A_shape), max_rank])
    row_inds = []
    col_inds = []
    uu = []
    vv = []

    def R_get_row(ii: int) -> float:
        return A_get_row(ii).reshape(-1) - np.sum([u[ii] * v for u, v in zip(uu, vv)], axis=0)

    def R_get_col(jj: int) -> float:
        return A_get_col(jj).reshape(-1) - np.sum([u * v[jj] for u, v in zip(uu, vv)], axis=0)

    norm_Ak_squared = 0.0
    ii = np.random.randint(A_shape[0])  # 0
    num_successes = 0
    for k in range(max_rank):
        R_row_i = R_get_row(ii)
        jj = np.argmax(np.abs(R_row_i))
        delta = R_row_i[jj]
        if delta == 0.0:
            if len(row_inds) == np.min(A_shape) - 1:
                if display:
                    print('Matrix recovered')
                break
        else:
            R_col_j = R_get_col(jj)
            u = R_col_j.copy()
            v = R_row_i.copy() / delta
            norm_uk = np.linalg.norm(u)
            norm_vk = np.linalg.norm(v)

            uu.append(u)
            vv.append(v)
            row_inds.append(ii)
            col_inds.append(jj)
            norm_Ak_squared += np.sum([np.dot(u, uj) * np.dot(v, vj) for uj, vj in zip(uu, vv)])

        R_col_j_modified = R_col_j
        R_col_j_modified[row_inds] = 0.0
        ii = np.argmax(np.abs(R_col_j_modified))

        if display:
            relerr_estimate = norm_uk * norm_vk / np.sqrt(norm_Ak_squared)
            print('k=', k, ', relerr_estimate=', relerr_estimate, ', num_successes=', num_successes)

        if (norm_uk * norm_vk) ** 2 < rtol_squared * norm_Ak_squared:
            num_successes += 1
            if num_successes >= error_estimator_rolling_window_size:
                if display:
                    print('rtol=', rtol, ' achieved.')
                break
        else:
            num_successes = 0

    return row_inds, col_inds, uu, vv

# Experimental / not tested below here

# def aca_plus(
#         A_get_row: typ.Callable[[int], np.ndarray],
#         A_get_col: typ.Callable[[int], np.ndarray],
#         A_shape: typ.Tuple[int, int],
#         max_rank: int = 100,
#         rtol: float = 1e-2,
#         display=False
# ) -> typ.Tuple[typ.List[int],  # row_inds=[i1, i2, ..., ik]
#                typ.List[int],  # col_inds=[j1, j2, ...m jk]
#                typ.List[np.ndarray],  # uu = [u1, u2, ..., uk]
#                typ.List[np.ndarray]  # vvt= [v1, v2, ..., vk]
# ]:
#     '''Constructs low rank approximation of matrix A by sampling rows and columns.
#     A =approx= u1 @ v1.T + u2 @ v2.T + ... + uk @ vk.T
#
#     See https://tbenthompson.com/book/tdes/low_rank.html
#
#     See also Algorithm 4 (ACA with partial pivoting) on page 16 in:
#     Jonas Ballani and Daniel Kressner.
#     "Matrices with hierarchical low-rank structures".
#     https://sma.epfl.ch/~anchpcommon/publications/cime.pdf
#     '''
#     rtol_squared = rtol * rtol
#     max_rank = np.min([np.min(A_shape), max_rank])
#     row_inds = []
#     col_inds = []
#     uu = []
#     vv = []
#
#     def R_get_row(ii: int) -> float:
#         return A_get_row(ii).reshape(-1) - np.sum([u[ii] * v for u, v in zip(uu, vv)], axis=0)
#
#     def R_get_col(jj: int) -> float:
#         return A_get_col(jj).reshape(-1) - np.sum([u * v[jj] for u, v in zip(uu, vv)], axis=0)
#
#     remaining_rows = np.ones(A_shape[0])
#     remaining_cols = np.ones(A_shape[1])
#
#     norm_Ak_squared = 0.0
#     ii = np.random.randint(A_shape[0])
#     jj = np.random.randint(A_shape[1])
#     for k in range(max_rank):
#         R_row_ii = R_get_row(ii)
#         jj_star = np.argmax(np.abs(R_row_ii * remaining_cols))
#         delta_row = R_row_ii[jj_star]
#
#         R_col_jj = R_get_col(jj)
#         ii_star = np.argmax(np.abs(R_col_jj * remaining_rows))
#         delta_col = R_col_jj[ii_star]
#
#         print('delta_row=', delta_row, ', delta_col=', delta_col)
#
#         if np.abs(delta_row) > np.abs(delta_col):
#             jj = jj_star
#             delta = delta_row
#             R_col_jj = R_get_col(jj)
#             # print('row wins')
#         else:
#             ii = ii_star
#             delta = delta_col
#             R_row_ii = R_get_row(ii)
#             # print('col wins')
#
#         print('ii=', ii, ', jj=', jj, ', delta=', delta)
#
#         remaining_rows[ii] = 0.0
#         remaining_cols[jj] = 0.0
#         ii = np.random.randint(A_shape[0])
#         jj = np.random.randint(A_shape[1])
#
#         if delta == 0.0:
#             if len(row_inds) == np.min(A_shape) - 1:
#                 if display:
#                     print('Matrix recovered')
#                 break
#         else:
#             u = R_col_jj.copy()
#             v = R_row_ii.copy() / delta
#             norm_uk = np.linalg.norm(u)
#             norm_vk = np.linalg.norm(v)
#
#             uu.append(u)
#             vv.append(v)
#             row_inds.append(ii)
#             col_inds.append(jj)
#             norm_Ak_squared += np.sum([np.dot(u, uj) * np.dot(v, vj) for uj, vj in zip(uu, vv)])
#
#         if display:
#             relerr_estimate = norm_uk * norm_vk / np.sqrt(norm_Ak_squared)
#             print('k=', k, ', relerr_estimate=', relerr_estimate)
#
#         if (norm_uk * norm_vk) ** 2 < rtol_squared * norm_Ak_squared:
#             if display:
#                 print('rtol=', rtol, ' achieved.')
#             break
#
#     return row_inds, col_inds, uu, vv


# from maxvol import py_maxvol
#
# def aca_alternating_maxvol_fixed_rank(
#         A_get_rows: typ.Callable[[typ.List[int]], np.ndarray],
#         A_get_cols: typ.Callable[[typ.List[int]], np.ndarray],
#         A_shape: typ.Tuple[int, int],
#         rank: int = 10,
#         col_inds_initial_guess: typ.List[int] = None,
#         max_iter: int = 20,
#         display=False
# ) -> typ.Tuple[typ.List[int],  # row_inds=[i1, i2, ..., ik]
#                typ.List[int],  # col_inds=[j1, j2, ..., jk]
#                typ.List[np.ndarray],  # col matrix C
#                typ.List[np.ndarray],  # mid matrix U
#                typ.List[np.ndarray]  # row matrix R
# ]:
#     '''Constructs low rank approximation of matrix A by sampling rows and columns.
#     A =approx= u1 @ v1.T + u2 @ v2.T + ... + uk @ vk.T
#     '''
#     rank = np.min([np.min(A_shape), rank])
#
#     def _maxvol(X):
#         rank = np.min(X.shape)
#         return aca_full(X, rtol=0.0, display=False, max_rank=rank, min_rank=rank)[0]
#         # return list(np.sort(py_maxvol(X)[0]))
#
#     col_inds = list(np.arange(rank)) if col_inds_initial_guess is None else col_inds_initial_guess
#     C = A_get_cols(col_inds)
#
#     row_inds = _maxvol(C)
#     R = A_get_rows(row_inds)
#
#     for ii in range(max_iter):
#         col_inds_new = _maxvol(R.T)
#         if col_inds == col_inds_new:
#             break
#         else:
#             col_inds = col_inds_new
#
#         C = A_get_cols(col_inds)
#
#         row_inds_new = _maxvol(C)
#         if row_inds == row_inds_new:
#             break
#         else:
#             row_inds = row_inds_new
#
#         R = A_get_rows(row_inds)
#
#     if display:
#         print('num_iter=', ii)
#
#     U = C[row_inds, :]
#     return row_inds, col_inds, C, U, R
