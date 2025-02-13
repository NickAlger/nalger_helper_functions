import numpy as np
import jax.numpy as jnp
import jax
import typing as typ
import functools as ft


# jax.config.update("jax_enable_x64", True) # enable double precision

@jax.jit
def low_rank_to_full(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ]
) -> jnp.ndarray: # X @ Y, shape=(N,M)
    X, Y = base
    return X @ Y


def left_orthogonalize_low_rank(
        x: typ.Tuple[
            jnp.ndarray, # X
            jnp.ndarray, # Y
        ],
) -> typ.Tuple[
    jnp.ndarray, # Q
    jnp.ndarray, # Y2
]:
    X, Y = x

    # Q, ss, Vt = jnp.linalg.svd(X, full_matrices=False)
    # R = ss.reshape((-1,1)) * Vt

    # Q, R = jnp.linalg.qr(X, mode='reduced') # jax QR returns nans, even in double precision
    Q, R = np.linalg.qr(X, mode='reduced')  #

    Y2 = R @ Y
    return jnp.array(Q), Y2


def right_orthogonalize_low_rank(x):
    x2_T = left_orthogonalize_low_rank((x[1].T, x[0].T))
    return (x2_T[1].T, x2_T[0].T)


@jax.jit
def right_orthogonalize_low_rank(x):
    x2_T = left_orthogonalize_low_rank((x[1].T, x[0].T))
    return (x2_T[1].T, x2_T[0].T)


@jax.jit
def tangent_vector_to_full(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
) -> jnp.ndarray:
    X, Y = base
    dX, dY = perturbation
    return low_rank_to_full((dX, Y)) + low_rank_to_full((X, dY))


@jax.jit
def add_sequences(
        uu: typ.Sequence[jnp.ndarray],
        vv: typ.Sequence[jnp.ndarray],
) -> typ.Tuple[jnp.ndarray]: # ww = (u1+v1, u2+v2, ...)
    return tuple([u + v for u, v in zip(uu, vv)])


@jax.jit
def inner_product_of_sequences(
        uu: typ.Sequence[jnp.ndarray],
        vv: typ.Sequence[jnp.ndarray],
) -> jnp.ndarray: # scalar, shape=()
    return jnp.sum(jnp.array([jnp.sum(u * v) for u, v in zip(uu, vv)]))


@jax.jit
def sequence_norm(
        perturbation1: typ.Tuple[
            jnp.ndarray, # dX1, shape=(N,r)
            jnp.ndarray, # dY1, shape=(r,M)
        ],
) -> jnp.ndarray: # scalar, shape=()
    return jnp.sqrt(inner_product_of_sequences(perturbation1, perturbation1))


@jax.jit
def scale_sequence(
        uu: typ.Sequence[jnp.ndarray],
        c: jnp.ndarray, # scalar, shape=()
) -> typ.Tuple[jnp.ndarray]: # ww = (c*u1, c*u2, ...)
    return tuple([c*u for u in uu])


@jax.jit
def subtract_sequences(
        uu: typ.Sequence[jnp.ndarray],
        vv: typ.Sequence[jnp.ndarray],
) -> typ.Tuple[jnp.ndarray]: # ww = (u1-v1, u2-v2, ...)
    return tuple([u - v for u, v in zip(uu, vv)])


@jax.jit
def tangent_oblique_projection(
        left_orthogonal_base: typ.Tuple[
            jnp.ndarray,  # Q, shape=(N,r), Q^T Q = I
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX_perp, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # dX2, shape=(N,r), Q^T dX2 = 0
    jnp.ndarray,  # dY2, shape=(r,M)
]:
    '''(dX, dY) -> (dX_perp, dY2) such that dX Y + X dY = dX_perp Y + X dY2 and X^T dX_perp = 0. '''
    Q, Y = left_orthogonal_base
    dX, dY = perturbation
    C = Q.T @ dX
    dX_parallel = Q @ C
    dX_perp = dX - dX_parallel
    dY2 = dY + C @ Y
    standard_perturbation = (dX_perp, dY2)
    return standard_perturbation


@jax.jit
def tangent_orthogonal_projection(
        left_orthogonal_base: typ.Tuple[
            jnp.ndarray,  # Q, shape=(N,r), Q^T Q = I
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX_perp, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # dX2, shape=(N,r), Q^T dX2 = 0
    jnp.ndarray,  # dY2, shape=(r,M)
]: # orthogonally project dX so that X^T dX = 0. Note: dX Y + X dY does not equal dX_perp Y + X dY2
    Q, Y = left_orthogonal_base
    dX, dY = perturbation
    return Q @ (Q.T @ dX), dY


@jax.jit
def tangent_oblique_projection_transpose(
        left_orthogonal_base: typ.Tuple[
            jnp.ndarray,  # Q, shape=(N,r), Q^T Q = I
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r), Q^T dX2 = 0
            jnp.ndarray,  # dY, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # dX2, shape=(N,r)
    jnp.ndarray,  # dY2, shape=(r,M)
]: # Transpose of linear map p -> tangent_oblique_projection(x, p)
    Q, Y = left_orthogonal_base
    dX, dY = perturbation
    dY2 = dY
    dX2 = dX - Q @ (Q.T @ dX) + Q @ (dY @ Y.T)
    perturbation2 = (dX2, dY2)
    return perturbation2


@jax.jit
def make_tangent_mass_matrix_helper(
        left_orthogonal_base: typ.Tuple[
            jnp.ndarray,  # Q, shape=(N,r), Q^T Q = I
            jnp.ndarray,  # Y, shape=(r,M)
        ],
) -> jnp.ndarray: # mass_matrix_helper_matrix=YY^T, shape=(r,r)
    Q, Y = left_orthogonal_base
    return Y @ Y.T


@jax.jit
def apply_tangent_mass_matrix(
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX_perp, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
        mass_matrix_helper_matrix: jnp.ndarray,  # shape=(r,r)
) ->  typ.Tuple[
    jnp.ndarray,  # MdX_perp, shape=(N,r)
    jnp.ndarray,  # MdY, shape=(r,M)
]: # mass_matrix @ standard_perturbation1
    dX_perp, dY = perturbation
    MdY = dY
    MdX = dX_perp @ mass_matrix_helper_matrix
    return MdX, MdY


@jax.jit
def inner_product_of_tangent_vectors(
        standard_perturbation1: typ.Tuple[
            jnp.ndarray,  # dX1_perp, shape=(N,r)
            jnp.ndarray,  # dY1, shape=(r,M)
        ],
        standard_perturbation2: typ.Tuple[
            jnp.ndarray,  # dX2_perp, shape=(N,r)
            jnp.ndarray,  # dY2, shape=(r,M)
        ],
        inner_product_helper_matrix: jnp.ndarray, # shape=(r,r)
) -> jnp.ndarray: # scalar, shape=()
    return inner_product_of_sequences(apply_tangent_mass_matrix(standard_perturbation1, inner_product_helper_matrix), standard_perturbation2)


@jax.jit
def add_low_rank_matrices(
        AA: typ.Sequence[
            typ.Tuple[
                jnp.ndarray,  # Xk, shape=(N,rk)
                jnp.ndarray,  # Yk, shape=(rk,M)
            ], # Ak = Xk @ Yk
        ],
) -> typ.Tuple[
    jnp.ndarray,  # X, shape=(N, r1+r2+...)
    jnp.ndarray,  # Y, shape=(r1+r2+..., M)
]: # A1 + ... + An = X @ Y
    X = jnp.hstack([A[0] for A in AA])
    Y = jnp.vstack([A[1] for A in AA])
    return X, Y


@jax.jit
def scale_low_rank_matrix(
        A: typ.Tuple[
            jnp.ndarray,  # Xk, shape=(N,r)
            jnp.ndarray,  # Yk, shape=(r,M)
        ], # A = X @ Y
        c,
) -> typ.Tuple[
    jnp.ndarray,  # X, shape=(N,r)
    jnp.ndarray,  # Y, shape=(r,M)
]: # c * A = X @ (c*Y)
    X0, Y0 = A
    return X0, c*Y0


@jax.jit
def attached_tangent_vector_as_low_rank(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # Q2, shape=(N, 3*r),
    jnp.ndarray,  # Y2, shape=(3*r, M)
]: # X Y + dX Y + X dY
    X, Y = base
    dX, dY = perturbation
    return add_low_rank_matrices([(X, Y), (dX, Y), (X, dY)])


# @ft.partial(jax.jit, static_argnames=['rank'])
def retract_tangent_vector(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
        rank: int,
) -> typ.Tuple[
    jnp.ndarray,  # Q2, shape=(N,r), Q2^T Q2 = I
    jnp.ndarray,  # Y2, shape=(r,M)
]: # retracted_vector based on truncated SVD of dX Y + Y dX. Is left orthogonal, even if base is not
    X, Y = base
    if rank is None:
        rank = X.shape[1]
    bigX, bigY = attached_tangent_vector_as_low_rank(base, perturbation)

    # QX, RX = jnp.linalg.qr(bigX, mode='reduced') # <-- jax qr is buggy, returns nans even in double precision
    # QYT, RYT = jnp.linalg.qr(bigY.T, mode='reduced')
    # U, ss, Vt = jnp.linalg.svd(RX @ RYT.T, full_matrices=False)
    QX, RX = np.linalg.qr(bigX, mode='reduced')
    QYT, RYT = np.linalg.qr(bigY.T, mode='reduced')
    U, ss, Vt = np.linalg.svd(RX @ RYT.T, full_matrices=False)
    Q = QX @ U[:,:rank]
    Y2 = (ss[:rank].reshape(-1,1) * Vt[:rank,:]) @ (QYT.T)
    retracted_vector = (jnp.array(Q), jnp.array(Y2))
    return retracted_vector