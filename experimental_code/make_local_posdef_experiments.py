import numpy as np
import matplotlib.pyplot as plt

# Experimental
# Make local matrix positive semi-definite by:
#  1) breaking the matrix into pieces using a partition of unity of the matrix entries
#  2) making each piece of the matrix positive semi-definite
#  3) adding the semi-definite pieces back together

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

A = np.zeros((len(tt), len(tt)))
for ii in range(A.shape[0]):
    pp = tt - mu[ii]
    A[ii,:] = V[ii]*np.exp(-0.5 * pp**2 / Sigma[ii])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for ii in [0,200,400,800,999]:
    plt.plot(tt, A[ii,:])
plt.title('rows of A')

plt.subplot(1,2,2)
for jj in [0,200,400,800,999]:
    plt.plot(tt, A[:,jj])
plt.title('cols of A')

A_sym = 0.5 * (A + A.T)
ee, P = np.linalg.eigh(A_sym)
A_sym_plus = P @ np.diag(ee * (ee > 0)) @ P.T


#### Create partition of unity on matrix space, Psik_hat

# landmark points
p0 = 0.0
p1 = 2.0
p2 = 4.0
p3 = 6.0
p4 = 8.0
p5 = 10.0

# un-normalized partition functions
psi_func = lambda x,y,p: np.exp(-0.5 * np.linalg.norm(np.array([x,y]) - np.array([p,p]))**2/(1.0**2))
Psi0 = np.array([[psi_func(tt[ii], tt[jj], p0) for jj in range(len(tt))] for ii in range(len(tt))])
Psi1 = np.array([[psi_func(tt[ii], tt[jj], p1) for jj in range(len(tt))] for ii in range(len(tt))])
Psi2 = np.array([[psi_func(tt[ii], tt[jj], p2) for jj in range(len(tt))] for ii in range(len(tt))])
Psi3 = np.array([[psi_func(tt[ii], tt[jj], p3) for jj in range(len(tt))] for ii in range(len(tt))])
Psi4 = np.array([[psi_func(tt[ii], tt[jj], p4) for jj in range(len(tt))] for ii in range(len(tt))])
Psi5 = np.array([[psi_func(tt[ii], tt[jj], p5) for jj in range(len(tt))] for ii in range(len(tt))])
all_Psi = [Psi0, Psi1, Psi2, Psi3, Psi4, Psi5]

plt.figure(figsize=(12,5))
for ii, Psi in enumerate(all_Psi):
    plt.subplot(1,len(all_Psi),ii+1)
    plt.imshow(Psi)
    plt.title('Psi'+str(ii))

# partition of unity on matrix space
Psi_sum = Psi0 + Psi1 + Psi2 + Psi3 + Psi4 + Psi5
Psi0_hat = Psi0 / Psi_sum
Psi1_hat = Psi1 / Psi_sum
Psi2_hat = Psi2 / Psi_sum
Psi3_hat = Psi3 / Psi_sum
Psi4_hat = Psi4 / Psi_sum
Psi5_hat = Psi5 / Psi_sum
all_Psi_hat = [Psi0_hat, Psi1_hat, Psi2_hat, Psi3_hat, Psi4_hat, Psi5_hat]

plt.figure(figsize=(12,5))
for ii, Psi_hat in enumerate(all_Psi_hat):
    plt.subplot(1,len(all_Psi_hat),ii+1)
    plt.imshow(Psi_hat)
    plt.title('Psi'+str(ii)+'_hat')

#### Break symmetrized matrix, A_sym, into pieces using partition of unity.

A_sym0 = Psi0_hat * A_sym
A_sym1 = Psi1_hat * A_sym
A_sym2 = Psi2_hat * A_sym
A_sym3 = Psi3_hat * A_sym
A_sym4 = Psi4_hat * A_sym
A_sym5 = Psi5_hat * A_sym
all_A_sym = [A_sym0, A_sym1, A_sym2, A_sym3, A_sym4, A_sym5]

plt.figure(figsize=(12,5))
for ii, A_symk in enumerate(all_A_sym):
    plt.subplot(1,len(all_A_sym),ii+1)
    plt.imshow(A_symk)
    plt.title('A_sym'+str(ii))

#### Make each piece of matrix positive definite and add back together

ee0, P0 = np.linalg.eigh(A_sym0)
ee1, P1 = np.linalg.eigh(A_sym1)
ee2, P2 = np.linalg.eigh(A_sym2)
ee3, P3 = np.linalg.eigh(A_sym3)
ee4, P4 = np.linalg.eigh(A_sym4)
ee5, P5 = np.linalg.eigh(A_sym5)

plt.figure()
plt.plot(ee0)
plt.plot(ee1)
plt.plot(ee2)
plt.plot(ee3)
plt.plot(ee4)
plt.plot(ee5)
plt.title('eigenvalues of components of A_sym')
plt.legend(['A_sym0', 'A_sym1', 'A_sym2', 'A_sym3', 'A_sym4', 'A_sym5'])

A_sym0_plus = P0 @ np.diag(ee0 * (ee0 > 0)) @ P0.T
A_sym1_plus = P1 @ np.diag(ee1 * (ee1 > 0)) @ P1.T
A_sym2_plus = P2 @ np.diag(ee2 * (ee2 > 0)) @ P2.T
A_sym3_plus = P3 @ np.diag(ee3 * (ee3 > 0)) @ P3.T
A_sym4_plus = P4 @ np.diag(ee4 * (ee4 > 0)) @ P4.T
A_sym5_plus = P5 @ np.diag(ee5 * (ee5 > 0)) @ P5.T

A_sym_plus_tilde = A_sym0_plus + A_sym1_plus + A_sym2_plus + A_sym3_plus + A_sym4_plus + A_sym5_plus

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(A_sym_plus)
plt.title('A_sym_plus')
plt.subplot(1,3,2)
plt.imshow(A_sym_plus_tilde)
plt.title('A_sym_plus_tilde')
plt.subplot(1,3,3)
plt.imshow(A_sym_plus_tilde-A_sym_plus)
plt.title('A_sym_plus_tilde-A_sym_plus')

err = np.linalg.norm(A_sym_plus_tilde - A_sym_plus) / np.linalg.norm(A_sym_plus)
print('err=', err)

####

from dataclasses import dataclass
import typing as typ
import functools as ft

@dataclass(frozen=True)
class PatchDenseMatrix:
    num_rows: int
    num_cols: int
    patch_matrices: typ.Tuple[np.ndarray]
    patch_row_inds: typ.Tuple[typ.Tuple[int]]
    patch_col_inds: typ.Tuple[typ.Tuple[int]]

    def __post_init__(me):
        assert(len(me.patch_matrices) == me.num_patches)
        assert(len(me.patch_row_inds) == me.num_patches)
        assert(len(me.patch_col_inds) == me.num_patches)
        assert(me.num_rows >= 0)
        assert(me.num_cols >= 0)

        for M, rr, cc in zip(me.patch_matrices, me.patch_row_inds, me.patch_col_inds):
            assert(M.shape == (len(rr), len(cc)))

        for rr in me.patch_row_inds:
            assert(np.all(np.array(rr) >= 0))
            assert(np.all(np.array(rr) < me.num_rows))

        for cc in me.patch_col_inds:
            assert(np.all(np.array(cc) >= 0))
            assert(np.all(np.array(cc) < me.num_cols))

    @ft.cached_property
    def num_patches(me) -> int:
        return len(me.patch_matrices)

    def matvec(
            me,
            X: np.ndarray,  # shape=(me.num_cols, K) or (me.num_cols,)
    ) -> np.ndarray:        # shape=(me.num_rows, K) or (me.num_rows,)
        if len(X.shape) == 1: # only one vector to matvec
            return me.matvec(X.reshape((-1,1))).reshape(-1)

        assert(len(X.shape) == 2)
        K = X.shape[1]
        assert(X.shape == (me.num_cols, K))

        Y = np.zeros(me.num_rows, K)
        for M, rr, cc in zip(me.patch_matrices, me.patch_row_inds, me.patch_col_inds):
            Y[rr, :] += M @ X[cc, :]
        return Y

    def rmatvec(
            me,
            Y: np.ndarray,  # shape=(me.num_rows, K) or (me.num_rows,)
    ) -> np.ndarray:        # shape=(me.num_cols, K) or (me.num_cols,)
        if len(Y.shape) == 1: # only one vector to matvec
            return me.matvec(Y.reshape((-1,1))).reshape(-1)

        assert(len(Y.shape) == 2)
        K = Y.shape[1]
        assert(Y.shape == (me.num_rows, K))

        X = np.zeros(me.num_cols, K)
        for M, rr, cc in zip(me.patch_matrices, me.patch_row_inds, me.patch_col_inds):
            X[cc, :] += M.T @ Y[rr, :]
        return Y

    def to_dense(me) -> np.ndarray: # shape=(me.num_rows, me.num_cols)
        A = np.zeros((me.num_rows, me.num_cols))
        for M, rr, cc in zip(me.patch_matrices, me.patch_row_inds, me.patch_col_inds):
            A[np.ix_(rr, cc)] += M
        return A

    def get_patch_contribution(me, ind: int) -> np.ndarray: # shape=(me.num_rows, me.num_cols)
        Ai = np.zeros((me.num_rows, me.num_cols))
        Ai[np.ix_(me.patch_row_inds[ind], me.patch_col_inds[ind])] = me.patch_matrices[ind]
        return Ai

    def make_patches_positive_semidefinite(me) -> 'PatchDenseMatrix':
        MM_plus: typ.List[np.ndarray] = []
        for M in me.patch_matrices:
            ee, P = np.linalg.eigh(0.5 * (M + M.T))
            M_plus = P @ np.diag(ee * (ee > 0)) @ P.T
            MM_plus.append(M_plus)
        return PatchDenseMatrix(me.num_rows, me.num_cols, tuple(MM_plus), me.patch_row_inds, me.patch_col_inds)


def Gaussian_Psi_func(
        row_min: np.ndarray, # shape=(dr,)
        row_max: np.ndarray, # shape=(dr,)
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
    # return np.outer(row_factor, col_factor)


def make_matrix_partition_of_unity(
        row_coords: np.ndarray, # shape=(num_rows, dr)
        col_coords: np.ndarray, # shape=(num_cols, dc)
        patch_row_inds: typ.Sequence[typ.Sequence[int]],
        patch_col_inds: typ.Sequence[typ.Sequence[int]],
        length_scale_factor = 0.33,
        normalize: bool = True,
) -> PatchDenseMatrix:
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

    all_Psi_hat: typ.List[np.ndarray] = []
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
        all_Psi_hat.append(Psi_hat)

    return PatchDenseMatrix(num_rows, num_cols, tuple(all_Psi_hat), patch_row_inds, patch_col_inds)


def make_partitioned_matrix(
        get_matrix_block: typ.Callable[
            [
                typ.List[int], # row inds, len=nrow_block
                typ.List[int], # col_inds, len=ncol_block
            ],
            np.ndarray, # A[np.ix_(row_inds, col_inds)], shape=(nrow_block, ncol_block)
        ],
        partition_of_unity: PatchDenseMatrix,
) -> PatchDenseMatrix:
    masked_AA = []
    for rr, cc, mask in zip(
            partition_of_unity.patch_row_inds,
            partition_of_unity.patch_col_inds,
            partition_of_unity.patch_matrices
    ):
        masked_AA.append(get_matrix_block(list(rr), list(cc)) * mask)

    return PatchDenseMatrix(
        partition_of_unity.num_rows, partition_of_unity.num_cols,
        tuple(masked_AA),
        partition_of_unity.patch_row_inds, partition_of_unity.patch_col_inds,
    )

#

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

#

patch_inds = [
    list(np.arange(0,400, dtype=int)),
    list(np.arange(200,600, dtype=int)),
    list(np.arange(400,800, dtype=int)),
    list(np.arange(600,1000, dtype=int)),
]

# patch_inds = [
#     list(np.arange(0,300, dtype=int)),
#     list(np.arange(200,500, dtype=int)),
#     list(np.arange(400,700, dtype=int)),
#     list(np.arange(600,900, dtype=int)),
#     list(np.arange(800,1000, dtype=int)),
# ]

# patch_inds = [
#     list(np.arange(0,300, dtype=int)),
#     list(np.arange(100,400, dtype=int)),
#     list(np.arange(200,500, dtype=int)),
#     list(np.arange(300,600, dtype=int)),
#     list(np.arange(400,700, dtype=int)),
#     list(np.arange(500,800, dtype=int)),
#     list(np.arange(600,900, dtype=int)),
#     list(np.arange(700,1000, dtype=int)),
# ]

# patch_inds = [
#     list(np.arange(0,200, dtype=int)),
#     list(np.arange(100,300, dtype=int)),
#     list(np.arange(200,400, dtype=int)),
#     list(np.arange(300,500, dtype=int)),
#     list(np.arange(400,600, dtype=int)),
#     list(np.arange(500,700, dtype=int)),
#     list(np.arange(600,800, dtype=int)),
#     list(np.arange(700,900, dtype=int)),
#     list(np.arange(800,1000, dtype=int)),
# ]

row_coords = tt.reshape((-1,1))
col_coords = row_coords
patch_row_inds = patch_inds
patch_col_inds = patch_inds

all_Psi = make_matrix_partition_of_unity(
        row_coords,
        col_coords,
        patch_row_inds,
        patch_col_inds,
        normalize=False,
)

plt.figure(figsize=(12,5))
for ii in range(all_Psi.num_patches):
    plt.subplot(1,all_Psi.num_patches,ii+1)
    plt.imshow(all_Psi.get_patch_contribution(ii))
    plt.title('Psi'+str(ii))

all_Psi_hat = make_matrix_partition_of_unity(
        row_coords,
        col_coords,
        patch_row_inds,
        patch_col_inds,
        normalize=True,
)

plt.figure(figsize=(12,5))
for ii in range(all_Psi_hat.num_patches):
    plt.subplot(1,all_Psi_hat.num_patches,ii+1)
    plt.imshow(all_Psi_hat.get_patch_contribution(ii))
    plt.title('Psi_hat'+str(ii))

plt.figure()
plt.imshow(all_Psi_hat.to_dense())
plt.title('Sum of all Psi_hat')

#

A_partitioned = make_partitioned_matrix(get_A_block, all_Psi_hat)

A2 = A_partitioned.to_dense()

partition_error = np.linalg.norm(A2 - A) / np.linalg.norm(A)
print('partition_error=', partition_error)

#

Asym = 0.5 * (A + A.T)

get_Asym_block = lambda rr, cc: 0.5 * (get_A_block(rr,cc) + get_A_block(cc,rr).T)

Asym_partitioned = make_partitioned_matrix(get_Asym_block, all_Psi_hat)

Asym2 = Asym_partitioned.to_dense()

sym_partition_error = np.linalg.norm(Asym2 - Asym) / np.linalg.norm(Asym)
print('sym_partition_error=', sym_partition_error)

#

ee, P = np.linalg.eigh(Asym)
Asym_plus = P @ np.diag(ee * (ee > 0)) @ P.T

Asym_plus_partitioned = Asym_partitioned.make_patches_positive_semidefinite()
Asym_plus2 = Asym_plus_partitioned.to_dense()

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