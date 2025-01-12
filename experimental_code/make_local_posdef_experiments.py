import numpy as np
from dataclasses import dataclass
import typing as typ
import functools as ft
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.linalg as sla
import matplotlib.pyplot as plt

# Experimental
# Make local matrix positive semi-definite by:
#  1) breaking the matrix into pieces using a partition of unity of the matrix entries
#  2) making each piece of the matrix positive semi-definite
#  3) adding the semi-definite pieces back together

def method_vectorized_or_unvectorized(
        method: typ.Callable[
            [
                typ.Any, # obj
                np.ndarray, # X, shape=(M,k)
            ],
            np.ndarray, # shape=(N,k)
        ]
) -> typ.Callable[
    [
        np.ndarray, # shape=(M,k) or shape=(M,)
    ],
    np.ndarray, # shape=(N,k) or shape=(N,)
]:
    '''obj.method(X) expects M-by-k matrix X as input, and returns a N-by-k matrix as output.
    Users want to call obj.method(x) where x is length-M vector, and get length-N vector output.
    This wraps obj.method() to seamlessly do either of these, depending on the shape of the input.'''
    def method_vec_or_unvec(
            obj,
            X: np.ndarray, # shape=(M,k) or shape=(M,)
    ):
        if len(X.shape) == 1:
            return method(obj, X.reshape((-1,1))).reshape(-1)
        assert(len(X.shape) == 2)
        return method(obj, X)
    return method_vec_or_unvec

@dataclass(frozen=True)
class LowRankMatrix:
    left_factor: np.ndarray
    right_factor: np.ndarray

    def __post_init__(me):
        assert(me.left_factor.shape == (me.shape[0], me.rank))
        assert(me.right_factor.shape == (me.rank, me.shape[1]))

    @ft.cached_property
    def shape(me) -> typ.Tuple[int, int]:
        return (me.left_factor.shape[0], me.right_factor.shape[1])

    @ft.cached_property
    def rank(me) -> int:
        return me.left_factor.shape[1]

    def to_dense(me) -> np.ndarray:
        return me.left_factor @ me.right_factor

    @method_vectorized_or_unvectorized
    def matvec(
            me,
            X: np.ndarray,  # shape=(num_cols, K) or (num_cols,)
    ) -> np.ndarray:        # shape=(num_rows, K) or (num_rows,)
        assert(X.shape[0] == me.shape[1])
        return me.left_factor @ (me.right_factor @ X)

    @method_vectorized_or_unvectorized
    def rmatvec(
            me,
            Y: np.ndarray,  # shape=(num_rows, K) or (num_rows,)
    ) -> np.ndarray:  # shape=(num_cols, K) or (num_cols,)
        assert(Y.shape[0] == me.shape[0])
        return me.right_factor.T @ (me.left_factor.T @ Y)


def pointwise_multiply_low_rank_matrices(
        A: LowRankMatrix, # shape=(N,M)
        B: LowRankMatrix, # shape=(N,M)
        rtol: float = 1e-5,
        max_rank: int = 100,
) -> LowRankMatrix: # shape=(N,M)
    assert(A.shape == B.shape)
    # This can be done faster without going to dense. Hope to improve implement this improvement later
    return make_low_rank(A.to_dense() * B.to_dense(), rtol=rtol, max_rank=max_rank)

def pointwise_multiply_many_low_rank_matrices(
        AA: typ.Tuple[LowRankMatrix],
        BB: typ.Tuple[LowRankMatrix],
        low_rank_max_rank: int = 100,
        low_rank_rtol: float = 1e-5,
) -> typ.Tuple[LowRankMatrix]:
    assert(len(AA) == len(BB))
    for A, B in zip(AA, BB):
        assert(A.shape == B.shape)

    all_AB = [pointwise_multiply_low_rank_matrices(A, B, rtol=low_rank_rtol, max_rank=low_rank_max_rank)
              for A, B in zip(AA, BB)]
    return tuple(all_AB)

@dataclass(frozen=True)
class PatchMatrix:
    shape: typ.Tuple[int, int]
    patch_matrices: typ.Tuple # elms ideally implement shape, matvec(), rmatvec(), and to_dense()
    patch_row_inds: typ.Tuple[typ.Tuple[int, ...], ...]
    patch_col_inds: typ.Tuple[typ.Tuple[int, ...], ...]

    def __post_init__(me):
        assert(len(me.patch_matrices) == me.num_patches)
        assert(len(me.patch_row_inds) == me.num_patches)
        assert(len(me.patch_col_inds) == me.num_patches)
        assert(me.shape[0] >= 0)
        assert(me.shape[1] >= 0)

        for M, rr, cc in zip(me.patch_matrices, me.patch_row_inds, me.patch_col_inds):
            assert(M.shape == (len(rr), len(cc)))

        for rr in me.patch_row_inds:
            assert(np.all(np.array(rr) >= 0))
            assert(np.all(np.array(rr) < me.shape[0]))

        for cc in me.patch_col_inds:
            assert(np.all(np.array(cc) >= 0))
            assert(np.all(np.array(cc) < me.shape[1]))

    @ft.cached_property
    def num_patches(me) -> int:
        return len(me.patch_matrices)

    def to_dense(me) -> np.ndarray: # shape=(me.num_rows, me.num_cols)
        A = np.zeros(me.shape)
        for M, rr, cc in zip(me.patch_matrices, me.patch_row_inds, me.patch_col_inds):
            A[np.ix_(rr, cc)] += M.to_dense()
        return A

    def get_patch_contribution(me, ind: int) -> np.ndarray: # shape=(me.num_rows, me.num_cols)
        Ai = np.zeros(me.shape)
        Ai[np.ix_(me.patch_row_inds[ind], me.patch_col_inds[ind])] = me.patch_matrices[ind].to_dense()
        return Ai

    @method_vectorized_or_unvectorized
    def matvec(
            me,
            X: np.ndarray,  # shape=(me.num_cols, K) or (me.num_cols,)
    ) -> np.ndarray:  # shape=(me.num_rows, K) or (me.num_rows,)
        K = X.shape[1]
        assert (X.shape == (me.shape[1], K))

        Y = np.zeros((me.shape[0], K))
        for M, rr, cc in zip(
                me.patch_matrices,
                me.patch_row_inds,
                me.patch_col_inds
        ):
            Y[rr, :] += M.matvec(X[cc, :])
        return Y

    @method_vectorized_or_unvectorized
    def rmatvec(
            me,
            Y: np.ndarray,  # shape=(me.num_rows, K) or (me.num_rows,)
    ) -> np.ndarray:  # shape=(me.num_cols, K) or (me.num_cols,)
        K = Y.shape[1]
        assert (Y.shape == (me.shape[0], K))

        X = np.zeros((me.shape[1], K))
        for M, rr, cc in zip(
                me.patch_matrices,
                me.patch_row_inds,
                me.patch_col_inds
        ):
            X[cc, :] += M.rmatvec(Y[rr, :])
        return X

    @method_vectorized_or_unvectorized
    def rsolve(
            me,
            X: np.ndarray,  # shape=(me.num_cols, K) or (me.num_cols,)
    ) -> np.ndarray:  # shape=(me.num_rows, K) or (me.num_rows,)
        K = X.shape[1]
        assert (X.shape == (me.shape[1], K))

        Y = np.zeros((me.shape[0], K))
        for M, rr, cc in zip(
                me.patch_matrices,
                me.patch_row_inds,
                me.patch_col_inds
        ):
            Y[rr, :] += M.rsolve(X[cc, :])
        return Y

    @method_vectorized_or_unvectorized
    def solve(
            me,
            Y: np.ndarray,  # shape=(me.num_rows, K) or (me.num_rows,)
    ) -> np.ndarray:  # shape=(me.num_cols, K) or (me.num_cols,)
        K = Y.shape[1]
        assert (Y.shape == (me.shape[0], K))

        X = np.zeros((me.shape[1], K))
        for M, rr, cc in zip(
                me.patch_matrices,
                me.patch_row_inds,
                me.patch_col_inds
        ):
            X[cc, :] += M.solve(Y[rr, :])
        return X


def Gaussian_Psi_func(
        row_min: np.ndarray, # shape=(dr,) <-- box min in physical space
        row_max: np.ndarray, # shape=(dr,) <-- box
        col_min: np.ndarray, # shape=(dc,)
        col_max: np.ndarray, # shape=(dc,)
        length_scale_factor: float,
        yy_row: np.ndarray, # shape=(nr, dr)
        xx_col: np.ndarray, # shape=(nc, dc)
) -> np.ndarray: # shape=(nr, nc)
    dr = len(row_min)
    dc = len(col_min)
    nr = yy_row.shape[0]
    nc = xx_col.shape[0]
    assert(row_min.shape == (dr,))
    assert(row_max.shape == (dr,))
    assert(col_min.shape == (dc,))
    assert(col_max.shape == (dc,))
    assert(yy_row.shape == (nr, dr))
    assert(xx_col.shape == (nc, dc))
    row_centroid = (row_max + row_min) / 2
    col_centroid = (col_max + col_min) / 2

    row_length_scales = length_scale_factor * (row_max - row_min) / 2
    col_length_scales = length_scale_factor * (col_max - col_min) / 2

    pp_row = (yy_row - row_centroid.reshape(1, dr)) / row_length_scales.reshape((1, dr))
    pp_col = (xx_col - col_centroid.reshape(1, dc)) / col_length_scales.reshape((1, dc))
    row_factor = np.exp(-0.5 * np.sum(pp_row**2, axis=1))
    col_factor = np.exp(-0.5 * np.sum(pp_col**2, axis=1))
    row_mask = np.logical_and(
        np.all(row_min.reshape((1,-1)) <= yy_row, axis=1),
        np.all(yy_row <= row_max.reshape((1,-1)), axis=1)
    )
    col_mask = np.logical_and(
        np.all(col_min.reshape((1, -1)) <= xx_col, axis=1),
        np.all(xx_col <= col_max.reshape((1, -1)), axis=1)
    )

    return np.outer(row_factor * row_mask, col_factor * col_mask)


def make_low_rank(
        A: np.ndarray, # shape=(N, M)
        rtol: float = 1e-5,
        max_rank: int = 100,
) -> LowRankMatrix:
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
    return LowRankMatrix(left_factor, right_factor)


def make_matrix_partition_of_unity(
        row_coords: np.ndarray, # shape=(num_rows, dr)
        col_coords: np.ndarray, # shape=(num_cols, dc)
        patch_row_inds: typ.Sequence[typ.Sequence[int]],
        patch_col_inds: typ.Sequence[typ.Sequence[int]],
        length_scale_factor = 0.33,
        normalize: bool = True,
        low_rank_rtol: float = 1e-4,
        low_rank_max_rank: int = 100,
) -> typ.Tuple[LowRankMatrix]:
    patch_row_inds = tuple([tuple(x) for x in patch_row_inds])
    patch_col_inds = tuple([tuple(x) for x in patch_col_inds])

    num_rows, dr = row_coords.shape
    num_cols, dc = col_coords.shape
    num_patches = len(patch_row_inds)
    assert(row_coords.shape == (num_rows, dr))
    assert(col_coords.shape == (num_cols, dc))
    assert(len(patch_row_inds) == num_patches)
    assert(len(patch_col_inds) == num_patches)

    for rr in patch_row_inds:
        assert(np.all(np.array(rr) >= 0))
        assert(np.all(np.array(rr) < num_rows))

    for cc in patch_col_inds:
        assert(np.all(np.array(cc) >= 0))
        assert(np.all(np.array(cc) < num_cols))

    patch_row_maxes = [np.max(row_coords[rr, :], axis=0) for rr in patch_row_inds]
    patch_row_mins  = [np.min(row_coords[rr, :], axis=0) for rr in patch_row_inds]

    patch_col_maxes = [np.max(col_coords[cc, :], axis=0) for cc in patch_col_inds]
    patch_col_mins  = [np.min(col_coords[cc, :], axis=0) for cc in patch_col_inds]

    all_Psi_hat: typ.List[LowRankMatrix] = []
    for ii in range(num_patches):
        patch_nr = len(patch_row_inds[ii])
        patch_nc = len(patch_col_inds[ii])
        Psi_hat = np.zeros((patch_nr, patch_nc))
        Psi_sum = np.zeros((patch_nr, patch_nc))
        for jj in range(num_patches):
            rows_overlap = (np.all(patch_row_mins[ii] <= patch_row_maxes[jj]) and
                            np.all(patch_row_mins[jj] <= patch_row_maxes[ii]))

            cols_overlap = (np.all(patch_col_mins[ii] <= patch_col_maxes[jj]) and
                            np.all(patch_col_mins[jj] <= patch_col_maxes[ii]))

            patches_overlap = (rows_overlap and cols_overlap)

            if patches_overlap or (ii==jj): # ii==jj condition shouldn't be necessary
                Psi_ij = Gaussian_Psi_func(
                    patch_row_mins[jj], patch_row_maxes[jj],
                    patch_col_mins[jj], patch_col_maxes[jj],
                    length_scale_factor,
                    row_coords[patch_row_inds[ii], :],
                    col_coords[patch_col_inds[ii], :],
                )
                Psi_sum += Psi_ij
                if ii == jj:
                    Psi_hat = Psi_ij
        if normalize:
            Psi_hat = Psi_hat / Psi_sum
        all_Psi_hat.append(
            make_low_rank(Psi_hat, rtol=low_rank_rtol, max_rank=low_rank_max_rank)
        )

    return tuple(all_Psi_hat)


def make_low_rank_matrix_positive_semidefinite(
        A: LowRankMatrix, # shape=(N,N)
        low_rank_rtol: float = 1e-5,
        low_rank_max_rank: int = 100,
        use_abs: bool = False,
) -> LowRankMatrix: # shape=(N,N)
    assert(A.shape[0] == A.shape[1])
    # This can be done more efficiently without going to dense. Hope to implement this improvement later
    A_dense = A.to_dense()
    ee, P = np.linalg.eigh(0.5 * (A_dense + A_dense.T))
    if use_abs:
        A_plus_dense = P @ np.diag(np.abs(ee)) @ P.T # take absolute value of eigs
    else:
        A_plus_dense = P @ np.diag(ee * (ee > 0)) @ P.T # threshold eigs at zero
    A_plus = make_low_rank(A_plus_dense, rtol=low_rank_rtol, max_rank=low_rank_max_rank)
    return A_plus


def make_low_rank_matrices_positive_semidefinite(
        matrices: typ.Sequence[LowRankMatrix],
        low_rank_max_rank: int = 100,
        low_rank_rtol: float = 1e-5,
) -> typ.Tuple[LowRankMatrix]:
    return tuple([make_low_rank_matrix_positive_semidefinite(
        A, low_rank_rtol=low_rank_rtol, low_rank_max_rank=low_rank_max_rank,
    ) for A in matrices])


# Tu = convolve(phi, u)
# T(y,x) = phi(y-x)
# T( . , x) = phi( . - x) = translate phi

def partition_matrix(
        get_matrix_block: typ.Callable[
            [
                typ.List[int], # row inds, len=nrow_block
                typ.List[int], # col_inds, len=ncol_block
            ],
            np.ndarray, # A[np.ix_(row_inds, col_inds)], shape=(nrow_block, ncol_block)
        ],
        patch_row_inds: typ.Sequence[typ.Sequence[int]],
        patch_col_inds: typ.Sequence[typ.Sequence[int]],
        low_rank_max_rank: int = 100,
        low_rank_rtol: float = 1e-4,
) -> typ.Tuple[LowRankMatrix]: # patch_matrices
    patch_matrices: typ.List[LowRankMatrix] = []
    for rr, cc in zip(
            patch_row_inds,
            patch_col_inds
    ):
        A = get_matrix_block(list(rr), list(cc))
        patch_matrices.append(
            make_low_rank(A, rtol=low_rank_rtol, max_rank=low_rank_max_rank)
        )

    return tuple(patch_matrices)

#

@dataclass(frozen=True)
class SparsePlusLowRankMatrix:
    sparse_matrix:  sps.csr_matrix # shape=(N,M)
    left_factor:    np.ndarray # shape=(N,r)
    right_factor:   np.ndarray # shape=(r,M)

    def __post_init__(me):
        assert(me.sparse_matrix.shape == me.shape)
        assert(me.left_factor.shape == (me.shape[0], me.LR_rank))
        assert(me.right_factor.shape == (me.LR_rank, me.shape[1]))

    @ft.cached_property
    def shape(me) -> typ.Tuple[int, int]:
        return me.sparse_matrix.shape

    @ft.cached_property
    def LR_rank(me) -> int:
        return me.left_factor.shape[1]

    @ft.cached_property
    def sparse_matrix_solve(me) -> typ.Callable:
        return spla.factorized(me.sparse_matrix)

    @ft.cached_property
    def capacitance_matrix(me) -> np.ndarray: # shape=(r,r)
        '''Computes matrix (I + R A^-1 L) from woodbury formula:
        (A + LR)^-1 = A^-1 - A^-1 L (I + R A^-1 L)^-1 R A^-1
        '''
        assert(me.shape[0] == me.shape[1])
        N = me.shape[0]
        r = me.LR_rank
        M1 = np.zeros((N, r))
        for ii in range(r):
            M1[:,ii] = me.sparse_matrix_solve(me.left_factor[:,ii])
        return np.eye(r) + me.right_factor @ M1

    @ft.cached_property
    def inv_capacitance_matrix(me) -> np.ndarray: # shape=(r,r)
        return np.linalg.inv(me.capacitance_matrix)

    def to_dense(me) -> np.ndarray:
        return me.sparse_matrix.toarray() + me.left_factor @ me.right_factor

    @method_vectorized_or_unvectorized
    def matvec(
            me,
            X: np.ndarray,  # shape=(num_cols, K) or (num_cols,)
    ) -> np.ndarray:        # shape=(num_rows, K) or (num_rows,)
        assert(X.shape[0] == me.shape[1])
        return me.sparse_matrix @ X + me.left_factor @ (me.right_factor @ X)

    @method_vectorized_or_unvectorized
    def rmatvec(
            me,
            Y: np.ndarray,  # shape=(num_rows, K) or (num_rows,)
    ) -> np.ndarray:  # shape=(num_cols, K) or (num_cols,)
        assert(Y.shape[0] == me.shape[0])
        return me.sparse_matrix.T @ Y + me.right_factor.T @ (me.left_factor.T @ Y)

    @method_vectorized_or_unvectorized
    def solve(
            me,
            Y: np.ndarray,  # shape=(num_cols, K) or (num_cols,)
    ) -> np.ndarray:        # shape=(num_rows, K) or (num_rows,)
        '''Use Woodbury formula to solve (A + LR)X = Y
        (A + LR)^-1 = A^-1 (I - L (I + R A^-1 L)^-1 R A^-1)
        '''
        assert(me.shape[0] == me.shape[1])
        assert(Y.shape[0] == me.shape[0])
        N = me.shape[0]
        k = Y.shape[1]

        M1 = np.zeros((N, k))
        for ii in range(k):
            M1[:,ii] = me.sparse_matrix_solve(Y[:,ii])
        M2 = Y - me.left_factor @ (me.inv_capacitance_matrix @ (me.right_factor @ M1))

        X = np.zeros((N,k))
        for ii in range(k):
            X[:,ii] = me.sparse_matrix_solve(M2[:,ii])

        return X

    @method_vectorized_or_unvectorized
    def rsolve(
            me,
            Y: np.ndarray,  # shape=(num_cols, K) or (num_cols,)
    ) -> np.ndarray:        # shape=(num_rows, K) or (num_rows,)
        '''Use Woodbury formula to solve (A + LR)^T X = Y
        (A + LR)^-T = A^-T (I - R^T (I + R A^-1 L)^-T L^T A^-T)
        '''
        assert(me.shape[0] == me.shape[1])
        assert(Y.shape[0] == me.shape[0])
        N = me.shape[0]
        k = Y.shape[1]

        M1 = np.zeros((N, k))
        for ii in range(k):
            M1[:,ii] = me.sparse_matrix_solve(Y[:,ii], trans='T')
        M2 = Y - me.right_factor.T @ (me.inv_capacitance_matrix.T @ (me.left_factor.T @ M1))

        X = np.zeros((N,k))
        for ii in range(k):
            X[:,ii] = me.sparse_matrix_solve(M2[:,ii], trans='T')

        return X


def partition_sparse_matrix(
        M: sps.csr_matrix,
        patch_row_inds: typ.Sequence[typ.Sequence[int]],
        patch_col_inds: typ.Sequence[typ.Sequence[int]],
) -> typ.Tuple[sps.csr_matrix]: # patch_matrices
    patch_matrices: typ.List[sps.csr_matrix] = []
    for rr, cc in zip(
            patch_row_inds,
            patch_col_inds
    ):
        Mi = ((M[rr,:].tocsc())[:,cc]).tocsr()
        patch_matrices.append(Mi)
    return tuple(patch_matrices)


def apply_weighted_linop(
        linop: typ.Callable[
            [
                np.ndarray, # shape=(M,k)
             ],
            np.ndarray, # shape=(N,k)
        ],
        weight_left_factor: np.ndarray, # shape=(N, rw)
        weight_right_factor: np.ndarray, # shape=(rw, M)
        X: np.ndarray, # shape=(M,k)
) -> np.ndarray: # # shape=(N,k)
    N, rw = weight_left_factor.shape
    M = weight_right_factor.shape[1]
    k = X.shape[1]
    assert(weight_right_factor.shape == (rw, M))
    assert(X.shape == (M,k))

    Y = np.zeros((N,k))
    for ii in range(rw):
        Y += weight_left_factor[:,ii].reshape((N,1)) * linop(weight_right_factor[ii,:].reshape((M,1)) * X)
    return Y


@dataclass(frozen=True)
class WeightedSparsePlusLowRankMatrix:
    matrix: SparsePlusLowRankMatrix # shape=(N,M)
    weights: LowRankMatrix # shape=(N,M)

    def __post_init__(me):
        assert(me.matrix.shape == me.shape)
        assert(me.weights.shape == me.shape)

    @ft.cached_property
    def shape(me) -> typ.Tuple[int, int]:
        return me.matrix.shape

    def to_dense(me) -> np.ndarray: # shape=me.shape
        return me.matrix.to_dense() * me.weights.to_dense()

    @method_vectorized_or_unvectorized
    def matvec(
            me,
            X: np.ndarray, # shape=(M,k) or shape=(M,)
    ) -> np.ndarray: # shape=(N,k) or shape=(N,)
        assert(X.shape[0] == me.shape[1])
        return apply_weighted_linop(me.matrix.matvec, me.weights.left_factor, me.weights.right_factor, X)

    @method_vectorized_or_unvectorized
    def rmatvec(
            me,
            Y: np.ndarray, # shape=(N,k) or shape=(N,)
    ) -> np.ndarray: # shape=(M,k) or shape=(M,)
        assert(Y.shape[0] == me.shape[0])
        return apply_weighted_linop(me.matrix.rmatvec, me.weights.right_factor.T, me.weights.left_factor.T, Y)

    @method_vectorized_or_unvectorized
    def solve(
            me,
            Y: np.ndarray, # shape=(N,k) or shape=(N,)
    ) -> np.ndarray: # shape=(N,k) or shape=(N,)
        assert(me.shape[0] == me.shape[1])
        assert(Y.shape[0] == me.shape[1])
        return apply_weighted_linop(me.matrix.solve, me.weights.left_factor, me.weights.right_factor, Y)

    @method_vectorized_or_unvectorized
    def rsolve(
            me,
            X: np.ndarray, # shape=(N,k) or shape=(N,)
    ) -> np.ndarray: # shape=(N,k) or shape=(N,)
        assert(me.shape[0] == me.shape[1])
        assert(X.shape[0] == me.shape[0])
        return apply_weighted_linop(me.matrix.rsolve, me.weights.right_factor.T, me.weights.left_factor.T, X)



#### Create spatially varying Gaussian kernel matrix, 'A', in 1D

tt = np.linspace(0, 10, 1000)

V = np.sqrt(1.0 - tt/1.5 + tt**2/3.1 - tt**3/50) # spatially varying volume
mu = tt + 0.1 * np.sin(5*tt) # spatially varying mean
Sigma = (0.5 + (10-tt)/30)**2

plt.figure()
plt.plot(tt, V)
plt.plot(tt, mu)
plt.plot(tt, Sigma)
plt.title('spatially varying moments')
plt.legend(['V', 'mu', 'Sigma'])

def get_A_block(rr, cc):
    block = np.zeros((len(rr), len(cc)))
    for i in range(len(rr)):
        r = rr[i]
        pp = tt[list(cc)] - mu[r]
        block[i,:] = V[r]*np.exp(-0.5 * pp**2 / Sigma[r])
    return block

all_inds = np.arange(len(tt), dtype=int)
A = get_A_block(all_inds, all_inds)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for ii in [0,200,400,800,999]:
    plt.plot(tt, A[ii,:])
plt.title('rows of A')

plt.subplot(1,2,2)
for jj in [0,200,400,800,999]:
    plt.plot(tt, A[:,jj])
plt.title('cols of A')


#### Create partition of unity on matrix space,

patch_inds = (
    tuple(list(np.arange(0,400, dtype=int))),
    tuple(list(np.arange(200,600, dtype=int))),
    tuple(list(np.arange(400,800, dtype=int))),
    tuple(list(np.arange(600,1000, dtype=int))),
)

row_coords = tt.reshape((-1,1))
col_coords = row_coords
patch_row_inds = patch_inds
patch_col_inds = patch_inds

all_Psi = make_matrix_partition_of_unity(
        row_coords, # shape=(num_dofs, spatial_dim) physical coordinates, used for evaluating gaussian
        col_coords,
        patch_row_inds,
        patch_col_inds,
        normalize=False,
)

all_Psi_PLRM = PatchMatrix(
    (row_coords.shape[0], col_coords.shape[0]),
    all_Psi, patch_row_inds, patch_col_inds
)

plt.figure(figsize=(12,5))
for ii in range(all_Psi_PLRM.num_patches):
    plt.subplot(1,all_Psi_PLRM.num_patches,ii+1)
    plt.imshow(all_Psi_PLRM.get_patch_contribution(ii))
    plt.title('Psi'+str(ii))

all_Psi_hat = make_matrix_partition_of_unity(
        row_coords,
        col_coords,
        patch_row_inds,
        patch_col_inds,
        normalize=True,
        low_rank_max_rank=10,
)

# all_Psi_hat = make_low_rank_matrices_positive_semidefinite(all_Psi_hat)

all_Psi_hat_PLRM = PatchMatrix(
    (row_coords.shape[0], col_coords.shape[0]),
    all_Psi_hat, patch_row_inds, patch_col_inds
)

plt.figure(figsize=(12,5))
for ii in range(all_Psi_hat_PLRM.num_patches):
    plt.subplot(1,all_Psi_hat_PLRM.num_patches,ii+1)
    plt.imshow(all_Psi_hat_PLRM.get_patch_contribution(ii))
    plt.title('Psi_hat'+str(ii))

plt.figure()
plt.imshow(all_Psi_hat_PLRM.to_dense())
plt.title('Sum of all Psi_hat')

#### Break matrix, A, into pieces using partition of unity.

A_pieces = partition_matrix(get_A_block, patch_row_inds, patch_col_inds)
A_pieces_weighted = pointwise_multiply_many_low_rank_matrices(A_pieces, all_Psi_hat)
A_patchmatrix = PatchMatrix(
    (len(row_coords), len(col_coords)),
    A_pieces_weighted,
    patch_row_inds, patch_col_inds,
)

A2 = A_patchmatrix.to_dense()

partition_error = np.linalg.norm(A2 - A) / np.linalg.norm(A)
print('partition_error=', partition_error)

X = np.random.randn(A2.shape[1], 13)
Y = A2 @ X
Y2 = A_patchmatrix.matvec(X)
err_matvec = np.linalg.norm(Y - Y2) / np.linalg.norm(Y)
print('err_matvec=', err_matvec)

Y = np.random.randn(A2.shape[0], 11)
X = A2.T @ Y
X2 = A_patchmatrix.rmatvec(Y)
err_rmatvec = np.linalg.norm(X - X2) / np.linalg.norm(X)
print('err_rmatvec=', err_rmatvec)

#### Break symmetrized matrix, A_sym, into pieces using partition of unity.

Asym = 0.5 * (A + A.T)

get_Asym_block = lambda rr, cc: 0.5 * (get_A_block(rr,cc) + get_A_block(cc,rr).T)

Asym_pieces = partition_matrix(get_Asym_block, patch_row_inds, patch_col_inds)
Asym_pieces_weighted = pointwise_multiply_many_low_rank_matrices(Asym_pieces, all_Psi_hat)
Asym_patchmatrix = PatchMatrix(
    (len(row_coords), len(col_coords)),
    Asym_pieces_weighted,
    patch_row_inds, patch_col_inds,
)

Asym2 = Asym_patchmatrix.to_dense()

sym_partition_error = np.linalg.norm(Asym2 - Asym) / np.linalg.norm(Asym)
print('sym_partition_error=', sym_partition_error)

#### Make each piece of partitioned matrix positive definite independently

ee, P = np.linalg.eigh(Asym)
Asym_plus = P @ np.diag(ee * (ee > 0)) @ P.T

Asym_plus_pieces = make_low_rank_matrices_positive_semidefinite(Asym_pieces)

Asym_plus_pieces_weighted = pointwise_multiply_many_low_rank_matrices(Asym_plus_pieces, all_Psi_hat)
Asym_plus_pieces_weighted_plus = make_low_rank_matrices_positive_semidefinite(Asym_plus_pieces_weighted)
Asym_plus_patchmatrix = PatchMatrix(
    (len(row_coords), len(col_coords)),
    # Asym_plus_pieces_weighted_plus,
    Asym_plus_pieces_weighted,
    patch_row_inds, patch_col_inds,
)

Asym_plus2 = Asym_plus_patchmatrix.to_dense()

plus_partition_err = np.linalg.norm(Asym_plus2 - Asym_plus) / np.linalg.norm(Asym_plus)
print('plus_partition_err=', plus_partition_err)

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(Asym_plus)
plt.title('A_sym_plus')
plt.subplot(1,3,2)
plt.imshow(Asym_plus2)
plt.title('Asym_plus2')
plt.subplot(1,3,3)
plt.imshow(Asym_plus-Asym_plus2)
plt.title('Asym_plus-Asym_plus2')


#### Sparse plus low rank stuff

N = len(tt)
L = sps.diags([-np.ones(N-1), 2*np.ones(N), -np.ones(N-1)], [-1,0,1], shape=(N,N)).tocsr()

xi = np.random.randn(N)
eta = spla.spsolve(L, xi)
plt.figure()
plt.plot(eta)

r = 12
left_factor = np.random.randn(N,r)
right_factor = np.random.randn(r,N)
M = SparsePlusLowRankMatrix(L, left_factor, right_factor)

M_dense = M.to_dense()

y1 = M_dense @ xi
y2 = M.matvec(xi)
err_slr_matvec = np.linalg.norm(y2-y1) / np.linalg.norm(y1)
print('err_slr_matvec=', err_slr_matvec)

y1 = M_dense.T @ xi
y2 = M.rmatvec(xi)
err_slr_rmatvec = np.linalg.norm(y2-y1) / np.linalg.norm(y1)
print('err_slr_rmatvec=', err_slr_rmatvec)

z1 = np.linalg.solve(M_dense, xi)
z2 = M.solve(xi)
err_woodburysolve = np.linalg.norm(z2-z1) / np.linalg.norm(z1)
print('err_woodburysolve=', err_woodburysolve)

z1 = np.linalg.solve(M_dense.T, xi)
z2 = M.rsolve(xi)
err_woodburyrsolve = np.linalg.norm(z2-z1) / np.linalg.norm(z1)
print('err_woodburyrsolve=', err_woodburyrsolve)

#

rw = 3
W = LowRankMatrix(np.random.randn(N, rw), np.random.randn(rw, N))
MW = WeightedSparsePlusLowRankMatrix(M, W)

MW_dense = M.to_dense() * W.to_dense()

xi = np.random.randn(N)
y1 = MW_dense @ xi
y2 = MW.matvec(xi)
err_wslr_matvec = np.linalg.norm(y2-y1) / np.linalg.norm(y1)
print('err_wslr_matvec=', err_wslr_matvec)

y1 = MW_dense.T @ xi
y2 = MW.rmatvec(xi)
err_wslr_rmatvec = np.linalg.norm(y2-y1) / np.linalg.norm(y1)
print('err_wslr_rmatvec=', err_wslr_rmatvec)

iM_dense = np.linalg.inv(M.to_dense())
y1 = (W.to_dense() * iM_dense) @ xi
y2 = MW.solve(xi)
err_wslr_solve = np.linalg.norm(y2-y1) / np.linalg.norm(y1)
print('err_wslr_solve=', err_wslr_solve)

y1 = (W.to_dense() * iM_dense).T @ xi
y2 = MW.rsolve(xi)
err_wslr_rsolve = np.linalg.norm(y2-y1) / np.linalg.norm(y1)
print('err_wslr_rsolve=', err_wslr_rsolve)

#
a = 1e-2
aL = a * L

L_pieces = partition_sparse_matrix(aL, patch_row_inds, patch_col_inds)
LA_pieces = tuple([SparsePlusLowRankMatrix(Li, X.left_factor, X.right_factor) for Li, X in zip(L_pieces, Asym_plus_pieces)])
WLA_pieces = tuple([WeightedSparsePlusLowRankMatrix(LA, PSI) for LA, PSI in zip(LA_pieces, all_Psi_hat)])
WLA_patchmatrix = PatchMatrix((N,N), WLA_pieces, patch_row_inds, patch_col_inds)

WLA_dense = WLA_patchmatrix.to_dense()
LA_dense = aL.toarray() + Asym_plus

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(WLA_dense)
plt.title('WLA_dense')
plt.subplot(1,3,2)
plt.imshow(LA_dense)
plt.title('LA_dense')
plt.subplot(1,3,3)
plt.imshow(LA_dense - WLA_dense)
plt.title('LA_dense - WLA_dense')

#

xi = np.random.randn(N)
eta1 = aL @ xi + Asym_plus_patchmatrix.matvec(xi)
eta2 = WLA_patchmatrix.matvec(xi)
err_wslr_patchmatrix_matvec = np.linalg.norm(eta2 - eta1) / np.linalg.norm(eta1)
print('err_wslr_patchmatrix_matvec=', err_wslr_patchmatrix_matvec)

eta1 = aL.T @ xi + Asym_plus_patchmatrix.rmatvec(xi)
eta2 = WLA_patchmatrix.rmatvec(xi)
err_wslr_patchmatrix_rmatvec = np.linalg.norm(eta2 - eta1) / np.linalg.norm(eta1)
print('err_wslr_patchmatrix_rmatvec=', err_wslr_patchmatrix_rmatvec)

eta1 = np.linalg.solve(aL.toarray() + Asym_plus, xi)
eta2 = WLA_patchmatrix.solve(xi)
err_wslr_patchmatrix_solve = np.linalg.norm(eta2 - eta1) / np.linalg.norm(eta1)
print('err_wslr_patchmatrix_solve=', err_wslr_patchmatrix_solve)

plt.figure()
plt.plot(eta1)
plt.plot(eta2)

#

iWLA_patchmatrix_dense = np.zeros((N,N))
for ii in range(N):
    ei = np.zeros(N)
    ei[ii] = 1.0
    iWLA_patchmatrix_dense[:,ii] = WLA_patchmatrix.solve(ei)

WLA_dense = aL.toarray() + Asym_plus
iWLA_dense = np.linalg.inv(WLA_dense)

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(iWLA_dense)
plt.title('iWLA_dense')
plt.subplot(1,3,2)
plt.imshow(iWLA_patchmatrix_dense)
plt.title('iWLA_patchmatrix_dense')
plt.subplot(1,3,3)
plt.imshow(iWLA_dense - iWLA_patchmatrix_dense)
plt.title('iWLA_dense - iWLA_patchmatrix_dense')

X = iWLA_patchmatrix_dense @ WLA_dense

U, ss, Vt = np.linalg.svd(X)
plt.figure()
plt.semilogy(ss)
plt.figure()
plt.plot(U[:,0])
plt.figure()
plt.plot(U[:,999])


ee, P = sla.eigh(iWLA_patchmatrix_dense, iWLA_dense)


