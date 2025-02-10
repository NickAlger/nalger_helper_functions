import numpy as np
import jax.numpy as jnp
import jax
import typing as typ
from functools import partial

import matplotlib.pyplot as plt


@jax.jit
def base_to_full(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ]
) -> jnp.ndarray: # XY, shape=(N,m)
    X, Y = base
    return X @ Y


@jax.jit
def left_orthogonalize_base(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray, # Q, shape=(N,r), Q^T Q = I
    jnp.ndarray, # Y2
]:
    '''QZ = XY. columns of Q are orthonormal'''
    X, Y = base
    Q, R = jnp.linalg.qr(X, mode='reduced')
    Y2 = R @ Y
    left_orthogonal_base = (Q, Y2)
    return left_orthogonal_base


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
    return base_to_full((dX, Y)) + base_to_full((X, dY))


@jax.jit
def add_tangent_vectors(
        perturbation1: typ.Tuple[
            jnp.ndarray, # dX1, shape=(N,r)
            jnp.ndarray, # dY1, shape=(r,M)
        ],
        perturbation2: typ.Tuple[
            jnp.ndarray,  # dX2, shape=(N,r)
            jnp.ndarray,  # dY2, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # dX1 + dX2, shape=(N,r)
    jnp.ndarray,  # dY1 + dY2, shape=(r,M)
]:
    return perturbation1[0] + perturbation2[0], perturbation1[1] + perturbation2[1]


@jax.jit
def dumb_inner_product(
        perturbation1: typ.Tuple[
            jnp.ndarray, # dX1, shape=(N,r)
            jnp.ndarray, # dY1, shape=(r,M)
        ],
        perturbation2: typ.Tuple[
            jnp.ndarray,  # dX2, shape=(N,r)
            jnp.ndarray,  # dY2, shape=(r,M)
        ],
) -> jnp.ndarray: # scalar, shape=()
    return jnp.sum(perturbation1[0] * perturbation2[0]) + jnp.sum(perturbation1[1] * perturbation2[1])


@jax.jit
def dumb_norm(
        perturbation1: typ.Tuple[
            jnp.ndarray, # dX1, shape=(N,r)
            jnp.ndarray, # dY1, shape=(r,M)
        ],
) -> jnp.ndarray: # scalar, shape=()
    return jnp.sqrt(dumb_inner_product(perturbation1, perturbation1))


@jax.jit
def scale_tangent_vector(
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
        c: jnp.ndarray, # scalar, shape=()
) -> typ.Tuple[
    jnp.ndarray,  # dX1, shape=(N,r)
    jnp.ndarray,  # c*dY1, shape=(r,M)
]: # scaled_perturbation
    dX, dY = perturbation
    scaled_perturbation = (c*dX, c*dY)
    return scaled_perturbation

@jax.jit
def subtract_tangent_vectors(
        perturbation1: typ.Tuple[
            jnp.ndarray, # dX1, shape=(N,r)
            jnp.ndarray, # dY1, shape=(r,M)
        ],
        perturbation2: typ.Tuple[
            jnp.ndarray,  # dX2, shape=(N,r)
            jnp.ndarray,  # dY2, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # dX1 + dX2, shape=(N,r)
    jnp.ndarray,  # dY1 + dY2, shape=(r,M)
]:
    return perturbation1[0] - perturbation2[0], perturbation1[1] - perturbation2[1]


@jax.jit
def standardize_perturbation(
        left_orthogonal_base: typ.Tuple[
            jnp.ndarray,  # Q, shape=(N,r), Q^T Q = I
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX_perp, shape=(N,r)
            jnp.ndarray,  # dY2, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # dX2, shape=(N,r), Q^T dX2 = 0
    jnp.ndarray,  # dY2, shape=(r,M)
]: # standard_perturbation
    Q, Y = left_orthogonal_base
    dX, dY = perturbation
    C = Q.T @ dX
    dX_parallel = Q @ C
    dX_perp = dX - dX_parallel
    dY2 = dY + C @ Y
    standard_perturbation = (dX_perp, dY2)
    return standard_perturbation


@jax.jit
def standardize_perturbation_transpose(
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
]: # perturbation2
    Q, Y = left_orthogonal_base
    dX, dY = perturbation
    dY2 = dY
    dX2 = dX - Q @ (Q.T @ dX) + Q @ (dY @ Y.T)
    perturbation2 = (dX2, dY2)
    return perturbation2


@jax.jit
def make_inner_product_helper_matrix(
        left_orthogonal_base: typ.Tuple[
            jnp.ndarray,  # Q, shape=(N,r), Q^T Q = I
            jnp.ndarray,  # Y, shape=(r,M)
        ],
) -> jnp.ndarray: # inner_product_helper_matrix=YY^T, shape=(r,r)
    Q, Y = left_orthogonal_base
    return Y @ Y.T


@jax.jit
def apply_tangent_mass_matrix(
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX_perp, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
        inner_product_helper_matrix: jnp.ndarray,  # shape=(r,r)
) ->  typ.Tuple[
    jnp.ndarray,  # MdX_perp, shape=(N,r)
    jnp.ndarray,  # MdY, shape=(r,M)
]: # mass_matrix @ standard_perturbation1
    dX_perp, dY = perturbation
    MdY = dY
    MdX = dX_perp @ inner_product_helper_matrix
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
    return dumb_inner_product(apply_tangent_mass_matrix(standard_perturbation1, inner_product_helper_matrix), standard_perturbation2)


@jax.jit
def tangent_vector_as_low_rank(
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
]: # big_base= X Y + dX Y + X dY
    X, Y = base
    dX, dY = perturbation
    bigX = jnp.hstack([X, X, dX])
    bigY = jnp.vstack([Y, dY, Y])
    big_base = (bigX, bigY)
    return big_base


@jax.jit
def retract_tangent_vector(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # Q2, shape=(N,r), Q2^T Q2 = I
    jnp.ndarray,  # Y2, shape=(r,M)
]: # retracted_vector based on truncated SVD of dX Y + Y dX. Is left orthogonal, even if base is not
    X, Y = base
    # dX, dY = perturbation
    r = X.shape[1]
    bigX, bigY = tangent_vector_as_low_rank(base, perturbation)

    # bigX = jnp.hstack([X, X, dX])
    # bigY = jnp.vstack([Y, dY, Y])
    QX, RX = jnp.linalg.qr(bigX, mode='reduced')
    QYT, RYT = jnp.linalg.qr(bigY.T, mode='reduced')
    U, ss, Vt = jnp.linalg.svd(RX @ RYT.T, full_matrices=False)
    Q = QX @ U[:,:r]
    Y2 = (ss[:r].reshape(-1,1) * Vt[:r,:]) @ (QYT.T)
    retracted_vector = (Q, Y2)
    return retracted_vector


#### Tests

jax.config.update("jax_enable_x64", True) # enable double precision

# Test base_to_full()

N = 17
r = 5
M = 12

X = np.random.randn(N,r)
Y = np.random.randn(r,M)
base = (X, Y)

A = base_to_full(base)
A_true = X @ Y
err_base_to_full = np.linalg.norm(A - A_true) / np.linalg.norm(A_true)
print('err_base_to_full=', err_base_to_full)

# test left_orthogonalize_base()

Q, Y2 = left_orthogonalize_base(base)

A2 = Q @ Y2
err_mult_base = np.linalg.norm(A2 - A_true) / np.linalg.norm(A_true)
print('err_mult_base=', err_mult_base)

err_orth_left_orthogonalize_base = np.linalg.norm(Q.T @ Q - np.eye(r))
print('err_orth_left_orthogonalize_base=', err_orth_left_orthogonalize_base)

# Test tangent_vector_to_full()

dX = np.random.randn(N,r)
dY = np.random.randn(r,M)
perturbation = (dX, dY)

s = 1e-7
v_diff = (base_to_full((X + s*dX, Y + s*dY)) - base_to_full((X, Y))) / s
v = tangent_vector_to_full(base, perturbation)
err_tangent_vector_to_full = np.linalg.norm(v - v_diff) / np.linalg.norm(v_diff)
print('s=', s, ', err_tangent_vector_to_full=', err_tangent_vector_to_full)

# Test add_tangent_vectors()

perturbation1 = perturbation

dX2 = np.random.randn(N,r)
dY2 = np.random.randn(r,M)
perturbation2 = (dX, dY)

perturbation12 = add_tangent_vectors(perturbation1, perturbation2)

v1 = tangent_vector_to_full(base, perturbation1)
v2 = tangent_vector_to_full(base, perturbation2)
v12_true = v1 + v2

v12 = tangent_vector_to_full(base, perturbation12)
err_add_tangent_vectors = np.linalg.norm(v12 - v12_true) / np.linalg.norm(v12_true)
print('err_add_tangent_vectors=', err_add_tangent_vectors)

# Test scale_tangent_vector()

c = np.random.randn()

scaled_perturbation = scale_tangent_vector(perturbation, c)

cv = tangent_vector_to_full(base, scaled_perturbation)
cv_true = c * tangent_vector_to_full(base, perturbation)
err_scale_tangent_vector = np.linalg.norm(cv - cv_true) / np.linalg.norm(cv_true)
print('err_scale_tangent_vector=', err_scale_tangent_vector)

# Test standardize_perturbation()

left_orthogonal_base = left_orthogonalize_base(base)
standard_perturbation = standardize_perturbation(left_orthogonal_base, perturbation)

v_true = tangent_vector_to_full(left_orthogonal_base, perturbation)
v = tangent_vector_to_full(left_orthogonal_base, standard_perturbation)
err_mult_standardize_perturbation = np.linalg.norm(v - v_true) / np.linalg.norm(v)
print('err_mult_standardize_perturbation=', err_mult_standardize_perturbation)

Q, _ = left_orthogonal_base
dX_perp, _ = standard_perturbation

err_perp_standardize_perturbation = np.linalg.norm(Q.T @ dX_perp)
print('err_perp_standardize_perturbation=', err_perp_standardize_perturbation)

# Test standardize_perturbation_transpose()

p = perturbation1
q = perturbation2
Fp = standardize_perturbation(left_orthogonal_base, p)
FTq = standardize_perturbation_transpose(left_orthogonal_base, q)

t1 = dumb_inner_product(Fp, q)
t2 = dumb_inner_product(p, FTq)

err_standardize_perturbation_transpose = np.abs(t1 - t2) / np.abs(t1 + t2)
print('err_standardize_perturbation_transpose=', err_standardize_perturbation_transpose)


# Test inner_product_of_tangent_vectors()

standard_perturbation1 = standardize_perturbation(left_orthogonal_base, perturbation1)
standard_perturbation2 = standardize_perturbation(left_orthogonal_base, perturbation2)

inner_product_helper_matrix = make_inner_product_helper_matrix(left_orthogonal_base)

IP = inner_product_of_tangent_vectors(standard_perturbation1, standard_perturbation2, inner_product_helper_matrix)

v1 = tangent_vector_to_full(left_orthogonal_base, standard_perturbation1)
v2 = tangent_vector_to_full(left_orthogonal_base, standard_perturbation2)
IP_true = np.sum(v1 * v2)

err_inner_product_of_tangent_vectors = np.linalg.norm(IP - IP_true) / np.linalg.norm(IP_true)
print('err_inner_product_of_tangent_vectors=', err_inner_product_of_tangent_vectors)

# Test tangent_vector_as_low_rank()

v1 = base_to_full(base) + tangent_vector_to_full(base, perturbation)
v2 = base_to_full(tangent_vector_as_low_rank(base, perturbation))

err_tangent_vector_as_low_rank = np.linalg.norm(v2-v1) / np.linalg.norm(v1)
print('err_tangent_vector_as_low_rank=', err_tangent_vector_as_low_rank)

# Test retract_tangent_vector()

retracted_vector = retract_tangent_vector(base, perturbation)
v = base_to_full(retracted_vector)

U, ss, Vt = np.linalg.svd(base_to_full(base) + tangent_vector_to_full(base, perturbation))
v_true = U[:,:r] @ np.diag(ss[:r]) @ Vt[:r,:]

err_retract_vector = np.linalg.norm(v - v_true) / np.linalg.norm(v_true)
print('err_retract_vector=', err_retract_vector)


######## Low rank matvecs objective function

@jax.jit
def forward_map(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        inputs: typ.Tuple[
            jnp.ndarray, # Omega, shape=(M,k)
            jnp.ndarray, # Omega_r, shape=(k_r,N)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # Z, shape=(N,k)
    jnp.ndarray,  # Z_r, shape=(k_r,M)
]: # outputs
    X, Y = base
    Omega, Omega_r = inputs
    Z = X @ (Y @ Omega)
    Z_r = Omega_r @ (X @ Y)
    outputs = Z, Z_r
    return outputs


@jax.jit
def objective(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        inputs: typ.Tuple[
            jnp.ndarray, # Omega, shape=(M,k)
            jnp.ndarray, # Omega_r, shape=(k_r,N)
        ],
        true_outputs: typ.Tuple[
            jnp.ndarray,  # Ytrue, shape=(N,k)
            jnp.ndarray,  # Ytrue_r, shape=(k_r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # J, scalar, shape=()
    typ.Tuple[
        jnp.ndarray,  # matvec residual norms squared. shape=(k)
        jnp.ndarray,  # rmatvec residual norms squared. shape=(k_r)
    ]
]:
    Ytrue, Ytrue_r = true_outputs
    Y, Y_r = forward_map(base, inputs)
    rsq = jnp.sum((Y - Ytrue)**2, axis=0)
    rsq_r = jnp.sum((Y_r - Ytrue_r)**2, axis=1)
    J = 0.5 * jnp.sum(rsq) + 0.5 * jnp.sum(rsq_r)
    return J, (rsq, rsq_r)


gradient_func = jax.jit(jax.grad(objective, argnums=0, has_aux=True))


@jax.jit
def forward_map_jvp(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
        inputs: typ.Tuple[
            jnp.ndarray, # Omega, shape=(M,k)
            jnp.ndarray, # Omega_r, shape=(k_r,N)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # Z, shape=(M,k)
    jnp.ndarray,  # Z_r, shape=(k_r,N)
]:
    X, Y = base
    # dX, dY = perturbation
    dX, dY = standardize_perturbation(base, perturbation)

    Omega, Omega_r = inputs
    Z = dX @ (Y @ Omega) + X @ (dY @ Omega)
    Z_r = (Omega_r @ dX) @ Y + (Omega_r @ X) @ dY
    return Z, Z_r


@jax.jit
def forward_map_vjp(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        inputs: typ.Tuple[
            jnp.ndarray, # Omega, shape=(M,k)
            jnp.ndarray, # Omega_r, shape=(k_r,N)
        ],
        ZZ: typ.Tuple[
            jnp.ndarray,  # Z, shape=(N,k)
            jnp.ndarray,  # Z_r, shape=(k_r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray, # shape=(N,r)
    jnp.ndarray, # shape=(r,M)
]:
    X, Y = base
    Z, Z_r = ZZ

    Omega, Omega_r = inputs
    dX = jnp.einsum('ix,aj,jx->ia', Z, Y, Omega) + jnp.einsum('xi,aj,xj->ia', Omega_r, Y, Z_r)
    dY = jnp.einsum('ix,ia,jx->aj', Z, X, Omega) + jnp.einsum('xi,ia,xj->aj', Omega_r, X, Z_r)

    # return dX, dY # <-- agrees with vjp autodiff
    return standardize_perturbation_transpose(base, (dX, dY)) # <-- agrees with vjp autodiff

    # X, Y = base
    # Z, Z_r = ZZ
    # Omega, Omega_r = inputs
    # dX = (Y.T @ Omega) @ Z + (Omega_r.T @ Z_r) @ Y.T
    # dY = (X.T @ Z) @ Omega.T + (Z_r @ Omega_r.T) @ X.T
    # return dX, dY

    # func = lambda b: forward_map(b, inputs)
    # _, vjp_func = jax.vjp(func, base)
    # return vjp_func(ZZ)[0]


@jax.jit
def gn_hessian_matvec(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
        inputs: typ.Tuple[
            jnp.ndarray, # Omega, shape=(M,k)
            jnp.ndarray, # Omega_r, shape=(k_r,N)
        ],
):
    return forward_map_vjp(base, inputs, forward_map_jvp(base, perturbation, inputs))


@jax.jit
def spd_sqrtm(A):
    ee, P = jnp.linalg.eigh(A)
    return P @ (jnp.sqrt(jnp.abs(ee)).reshape((-1,1)) * P.T)


@jax.jit
def tangent_space_objective(
        left_orthogonal_base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
        inputs: typ.Tuple[
            jnp.ndarray,  # Omega, shape=(M,k)
            jnp.ndarray,  # Omega_r, shape=(k_r,N)
        ],
        true_outputs: typ.Tuple[
            jnp.ndarray,  # Ztrue, shape=(N,k)
            jnp.ndarray,  # Ztrue_r, shape=(k_r,M)
        ],
):
    J0, _ = objective(left_orthogonal_base, inputs, true_outputs)

    M_helper = make_inner_product_helper_matrix(left_orthogonal_base)
    sqrtM_helper = spd_sqrtm(M_helper)
    isqrtM_helper = jnp.linalg.inv(sqrtM_helper)

    # p0 = standardize_perturbation(left_orthogonal_base, apply_tangent_mass_matrix(perturbation, isqrtM_helper))
    # p0 = standardize_perturbation(left_orthogonal_base, perturbation)
    p0 = perturbation
    p = p0
    # p = apply_tangent_mass_matrix(p0, isqrtM_helper)
    # p = apply_tangent_mass_matrix(p0, sqrtM_helper)

    g0, _ = gradient_func(left_orthogonal_base, inputs, true_outputs)
    # g = standardize_perturbation(left_orthogonal_base, g0)
    g = g0
    # g = apply_tangent_mass_matrix(g0, sqrtM_helper)

    # gp = inner_product_of_tangent_vectors(g, p, M_helper)
    gp = dumb_inner_product(g, p)

    Hp = gn_hessian_matvec(left_orthogonal_base, p, inputs)
    # Hp = standardize_perturbation_transpose(left_orthogonal_base, gn_hessian_matvec(left_orthogonal_base, p, inputs))

    # pHp = inner_product_of_tangent_vectors(p, Hp, M_helper)
    pHp = dumb_inner_product(p, Hp)

    return 0.5 * pHp + gp + J0


# Test

A0 = np.random.randn(20,13)
A = A0.T @ A0
sqrtA_true = jax.scipy.linalg.sqrtm(A)
sqrtA = spd_sqrtm(A)

err_spd_sqrtm = np.linalg.norm(sqrtA_true - sqrtA) / np.linalg.norm(sqrtA_true)
print('err_spd_sqrtm=', err_spd_sqrtm)

#

N = 100
M = 89
r = 5

U, _, Vt = np.linalg.svd(np.random.randn(N, M), full_matrices=False)
ss = np.logspace(-30, 0, np.minimum(N,M))
A = U @ np.diag(ss) @ Vt

Omega = jnp.array(np.random.randn(M, r+5))
Omega_r = jnp.array(np.random.randn(r+5, N))
Ytrue = A @ Omega
Ytrue_r = Omega_r @ A
inputs = (Omega, Omega_r)
true_outputs = (Ytrue, Ytrue_r)

X = jnp.array(np.random.randn(N, r))
Y = jnp.array(np.random.randn(r, M))
base = (X, Y)

Y2, Y2_r = forward_map(base, inputs)

A2 = base_to_full(base)
Y2true = A2 @ Omega
Y2true_r = Omega_r @ A2

err_forward_map = np.linalg.norm(Y2 - Y2true) / np.linalg.norm(Y2true)
err_forward_map_r = np.linalg.norm(Y2_r - Y2true_r) / np.linalg.norm(Y2true_r)

print('err_forward_map=', err_forward_map)
print('err_forward_map_r=', err_forward_map_r)

#

J, (rsq, rsq_r) = objective(base, inputs, true_outputs)

rsq_true = np.linalg.norm(Y2 - Ytrue, axis=0)**2
rsq_true_r = np.linalg.norm(Y2_r - Ytrue_r, axis=1)**2
Jtrue = 0.5 * np.linalg.norm(Y2 - Ytrue)**2 + 0.5 * np.linalg.norm(Y2_r - Ytrue_r)**2

err_rsq_objective = np.linalg.norm(rsq - rsq_true)
err_rsq_r_objective = np.linalg.norm(rsq_r - rsq_true_r)
err_J_objective = np.linalg.norm(J - Jtrue) / np.linalg.norm(Jtrue)
print('err_rsq_objective=', err_rsq_objective)
print('err_rsq_r_objective=', err_rsq_r_objective)
print('err_J_objective=', err_J_objective)

#

dX = np.random.randn(N, r)
dY = np.random.randn(r, M)
perturbation = (dX, dY)

df = forward_map_jvp(base, perturbation, inputs)

s = 1e-6
f = forward_map(base, inputs)
f2 = forward_map((base[0]+s*perturbation[0], base[1] + s*perturbation[1]), inputs)
df_diff = ((f2[0] - f[0]) / s, (f2[1] - f[1]) / s)

err_forward_map_jvp0 = np.linalg.norm(df[0] - df_diff[0]) / np.linalg.norm(df_diff[0])
print('s=', s, ', err_forward_map_jvp0=', err_forward_map_jvp0)

err_forward_map_jvp1 = np.linalg.norm(df[1] - df_diff[1]) / np.linalg.norm(df_diff[1])
print('s=', s, ', err_forward_map_jvp1=', err_forward_map_jvp1)

#

left_orthogonal_base = left_orthogonalize_base(base)
standard_perturbation = standardize_perturbation(left_orthogonal_base, perturbation)

Z = np.random.randn(*Ytrue.shape)
Z_r = np.random.randn(*Ytrue_r.shape)
ZZ = (Z, Z_r)

Jp = forward_map_jvp(left_orthogonal_base, standard_perturbation, inputs)
JtZ = forward_map_vjp(left_orthogonal_base, inputs, ZZ)

t1 = dumb_inner_product(Jp, ZZ) # np.sum(Jp[0] * ZZ[0]) + np.sum(Jp[1] * ZZ[1])
t2 = dumb_inner_product(JtZ, standard_perturbation) # np.sum(JtZ[0] * standard_perturbation[0]) + np.sum(JtZ[1] + standard_perturbation[1])

err_forward_map_vjp = np.abs(t1 - t2) / np.abs(t1 + t2)
print('err_forward_map_vjp=', err_forward_map_vjp)

#

J = tangent_space_objective(
    left_orthogonal_base, standard_perturbation, inputs, true_outputs
)

big_base = tangent_vector_as_low_rank(left_orthogonal_base, standard_perturbation)
J_true, _ = objective(big_base, inputs, true_outputs)

err_tangent_space_objective = np.abs(J - J_true) / np.abs(J_true)
print('err_tangent_space_objective=', err_tangent_space_objective)


#### CG Steihaug

def cg_steihaug(
        hessian_matvec:         typ.Callable[[typ.Any], typ.Any], # u -> H @ u
        gradient:               typ.Any,
        add:            typ.Callable[[typ.Any, typ.Any], typ.Any],                          # (u, v) -> u+v
        scale:          typ.Callable[[typ.Any, typ.Union[float, jnp.ndarray]], typ.Any],    # (u, c) -> c*u
        inner_product:  typ.Callable[[typ.Any, typ.Any], typ.Union[float, jnp.ndarray]],    # (u, v) -> <u,v>
        trust_region_radius:    typ.Union[float, jnp.ndarray],
        rtol:                   typ.Union[float, jnp.ndarray],
        max_iter:               int  = 250,
        display:                bool = True,
        callback:               typ.Callable = None,
) -> typ.Tuple[
    typ.Any, # optimal search direction. same type as gradient
    typ.Tuple[
        int, # number of iterations
        str, # termination reason
    ]
]:
    '''Algorithm 7.2 in Nocedal and Wright, page 171. See also: equation 4.5 on page 69.
    Approximately solves quadratic minimization problem:
        argmin_p f + g^T p + 0.5 * p^T H P
    '''
    def _print(s: str):
        if display:
            print(s)

    if callback is None:
        callback = lambda z: None

    gradnorm_squared = inner_product(gradient, gradient)
    atol_squared = rtol**2 * gradnorm_squared

    z = scale(gradient, 0.0)
    r = gradient
    d = scale(gradient, -1.0)

    z_normsquared = 0.0
    r_normsquared = inner_product(r, r)
    _print('initial ||r||/||g||=' + str(jnp.sqrt(r_normsquared / gradnorm_squared)))

    if r_normsquared < atol_squared:
        return z, (0, 'tolerance_achieved')  # currently zero; take no steps

    for jj in range(max_iter):
        Bd  = hessian_matvec(d)
        dBd = inner_product(d, Bd)
        if dBd <= 0.0:
            if np.isinf(trust_region_radius):
                if jj == 0:
                    p = gradient
                    _print('gradient points downhill. Probably due to numerical errors. dBd=' + str(dBd))
                    return p, (jj + 1, 'gradient_points_downhill')
                else:
                    p = z
                    _print('Iterate ' + str(jj) + ' encountered negative curvature. dBd=' + str(dBd))
                    return p, (jj + 1, 'encountered_negative_curvature')

            tau1, tau2 = _interpolate_to_trust_region_boundary(z, d, z_normsquared, trust_region_radius, inner_product, display=True)
            p1 = add(z, scale(d, tau1)) # p = z + tau1*d
            p2 = add(z, scale(d, tau2)) # p = z + tau2*d
            m1 = inner_product(gradient, p1) + 0.5 * inner_product(p1, hessian_matvec(p1))
            m2 = inner_product(gradient, p2) + 0.5 * inner_product(p2, hessian_matvec(p2))
            if m1 <= m2:
                tau = tau1
                p = p1
            else:
                tau = tau2
                p = p2
            _print('m1=' + str(m1) + ', m2=' + str(m2))
            _print('Iterate ' + str(jj) + ' encountered negative curvature. tau=' + str(tau))
            return p, (jj+1, 'encountered_negative_curvature')

        alpha = r_normsquared / dBd
        z_next = add(z, scale(d, alpha)) # z_next = z + alpha*d
        z_next_normsquared = inner_product(z_next, z_next)

        if z_next_normsquared >= trust_region_radius ** 2:
            tau, _ = _interpolate_to_trust_region_boundary(z, d, z_normsquared, trust_region_radius, inner_product, display)
            p = add(z, scale(d, tau)) # p = z + tau*d
            _print(
                'Iterate ' + str(jj)
                + ' exited trust region. trust_region_radius=' + str(trust_region_radius)
                + ', tau=' + str(tau)
            )
            return p, (jj+1, 'exited_trust_region')

        r_next = add(r, scale(Bd, alpha)) # r_next = r + alpha*Bd
        r_next_normsquared: jnp.ndarray = inner_product(r_next, r_next) # scalar, shape=()
        _print(
            'CG Iter:' + str(jj)
            + ', ||r||/||g||=' + str(jnp.sqrt(jnp.sqrt(r_next_normsquared / gradnorm_squared)))
            + ', ||z||=' + str(jnp.sqrt(z_next_normsquared))
            + ', Delta=' + str(trust_region_radius)
        )
        if r_next_normsquared < atol_squared:
            _print('rtol=' + str(rtol) + ' achieved. ||r||/||g||=' + str(jnp.sqrt(r_next_normsquared / gradnorm_squared)))
            callback(z_next)
            return z_next, (jj+1, 'tolerance_achieved')

        beta: jnp.ndarray = r_next_normsquared / r_normsquared # scalar, shape=()
        d = add(scale(r_next, -1.0), scale(d, beta))

        r = r_next
        z = z_next
        callback(z)
        z_normsquared = z_next_normsquared
        r_normsquared = r_next_normsquared

    _print('Reached max_iter=' + str(max_iter) + ' without converging or exiting trust region.')

    return z, (max_iter, 'performed_maximum_iterations')


def _interpolate_to_trust_region_boundary(
        z:                      typ.Any,
        d:                      typ.Any,
        z_normsquared:          jnp.ndarray, # scalar, shape=()
        trust_region_radius:    jnp.ndarray, # shape=()
        inner_product:          typ.Callable[[typ.Any, typ.Any], typ.Union[float, jnp.ndarray]],  # (u, v) -> <u,v>
        display:                bool,
) -> typ.Tuple[jnp.ndarray, jnp.ndarray]: # (tau1, tau2), scalars, elm_shape=()
    '''Find tau such that:
        Delta^2 = ||z + tau*d||^2
                = ||z||^2 + tau*2*(z, d) + tau^2*||d||^2
    which is:,
      tau^2 * ||d||^2 + tau * 2*(z, d) + ||z||^2-Delta^2 = 0
              |--a--|         |--b---|   |------c------|
    Note that c is negative since we were inside trust region last step
    '''
    d_normsquared = inner_product(d, d)
    z_dot_d = inner_product(z, d)

    a = d_normsquared
    b = 2.0 * z_dot_d
    c = z_normsquared - trust_region_radius ** 2

    tau1 = (-b + jnp.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)  # quadratic formula, first root
    tau2 = (-b - jnp.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)  # quadratic formula, second root
    if display:
        print('a=', a, ', b=', b, ', c=', c, ', tau1=', tau1, ', tau2=', tau2)
    return tau1, tau2


# Test cg_steihaug

N = 100
cond = 50

U, _, _ = np.linalg.svd(np.random.randn(N, N))
ss = 1 + np.logspace(0, np.log10(cond), N)
H = U @ np.diag(ss) @ U.T

hessian_matvec = lambda x: H @ x
gradient = np.random.randn(N)
add = lambda u,v: u+v
scale = lambda u,c: c*u
inner_product = lambda u,v: np.dot(u,v)

# Check that this generates the same iterates as reference CG from stackoverflow if:
#  - trust region is infinite,
#  - Hessian is positive definite,
#  - tolerance is zero

max_iter=25

pp = []
callback = lambda z: pp.append(z)

p, aux = cg_steihaug(
    hessian_matvec, gradient, add, scale, inner_product,
    trust_region_radius=np.inf, rtol=0.0, max_iter=max_iter, callback=callback,
)


# This basic implementation from stackoverflow agrees with me to machine precision.
# Scipy's CG doesn't seem to agree with me, although the two converge to each other.
# I think this has something to so with where the callback is called in the scipy implementation.
def basic_cg(A, b, x, num_iter):
    # https://stackoverflow.com/a/60847526/484944
    pp_so = []
    r = b - A.dot(x)
    p = r.copy()
    for i in range(num_iter):
        Ap = A.dot(p)
        alpha = np.dot(p, r) / np.dot(p, Ap)
        x = x + alpha * p
        pp_so.append(x)
        r = b - A.dot(x)
        beta = -np.dot(r, Ap) / np.dot(p, Ap)
        p = r + beta * p
    return pp_so

pp_so = basic_cg(H, -gradient, -0.0*gradient, max_iter)

err_CG_iterates = np.linalg.norm(np.array(pp) - np.array(pp_so), axis=1) / np.linalg.norm(np.array(pp_so), axis=1)
print('err_CG_iterates=', err_CG_iterates)

# Check that we stop when the tolerance is reached

for rtol in [1e-1, 1e-2, 1e-3]:
    p, aux = cg_steihaug(
        hessian_matvec, gradient, add, scale, inner_product,
        trust_region_radius=np.inf, rtol=rtol, max_iter=N,
    )
    relres = np.linalg.norm(H @ p + gradient) / np.linalg.norm(gradient)
    print('rtol=', rtol, ', relres=', relres)

# Check that we stop when we exit the trust region

norm_p_true = np.linalg.norm(np.linalg.solve(H, -gradient))

for scaling in [0.1, 0.5, 0.9, 0.99]:
    trust_radius = scaling * norm_p_true
    p, aux = cg_steihaug(
        hessian_matvec, gradient, add, scale, inner_product,
        trust_region_radius=trust_radius, rtol=0.0, max_iter=N,
    )
    norm_p = np.linalg.norm(p)
    print('trust_radius=', trust_radius, ', norm_p=', norm_p)


#### Use iterative CG-Steihaug to solve low rank fit problem

N = 100
M = 89
num_samples = 10

U, _, Vt = np.linalg.svd(np.random.randn(N, M), full_matrices=False)
ss = np.logspace(-30, 0, np.minimum(N,M))
A = U @ np.diag(ss) @ Vt

Omega = jnp.array(np.random.randn(M, num_samples))
Omega_r = jnp.array(np.random.randn(num_samples, N))
Ytrue = A @ Omega
Ytrue_r = Omega_r @ A
inputs = (Omega, Omega_r)
true_outputs = (Ytrue, Ytrue_r)

#

rank = 1

X = jnp.array(np.random.randn(N, rank))
Y = jnp.array(np.random.randn(rank, M))
base = (X, Y)
left_orthogonal_base = left_orthogonalize_base(base)

#

J, (relerrs, relerrs_r) = objective(left_orthogonal_base, inputs, true_outputs)
g0, _ = gradient_func(left_orthogonal_base, inputs, true_outputs)
g = standardize_perturbation(left_orthogonal_base, g0)
norm_g = dumb_norm(g)

print('J=', J)
print('relerrs=', relerrs)
print('relerrs_r=', relerrs_r)
print('||g||=', norm_g)

for _ in range(30):
    apply_H = lambda P: gn_hessian_matvec(left_orthogonal_base, P, inputs)

    M_helper = make_inner_product_helper_matrix(left_orthogonal_base)
    ip_func = lambda p1, p2: inner_product_of_tangent_vectors(
        standardize_perturbation(left_orthogonal_base, p1), standardize_perturbation(left_orthogonal_base, p2), M_helper
    )

    trust_radius = 1e6
    rtol = 0.05

    p, aux = cg_steihaug(
        apply_H, g, add_tangent_vectors, scale_tangent_vector, ip_func, trust_radius, rtol, display=True,
    )

    base = retract_tangent_vector(left_orthogonal_base, p)
    left_orthogonal_base = left_orthogonalize_base(base)

    J, (relerrs, relerrs_r) = objective(left_orthogonal_base, inputs, true_outputs)
    g0, _ = gradient_func(left_orthogonal_base, inputs, true_outputs)
    g = standardize_perturbation(left_orthogonal_base, g0)
    norm_g = dumb_norm(g)

    print('J=', J)
    print('relerrs=', np.sqrt(relerrs))
    print('relerrs_r=', np.sqrt(relerrs_r))
    print('||g||=', norm_g)

A2 = base_to_full(left_orthogonal_base)
computed_err = np.linalg.norm(A2 - A) / np.linalg.norm(A)
print('rank=', rank)
print('computed_err=', computed_err)

U, ss, Vt = np.linalg.svd(A)
Ar = U[:,:rank] @ np.diag(ss[:rank]) @ Vt[:rank,:]

ideal_err = np.linalg.norm(Ar - A) / np.linalg.norm(A)
print('ideal_err=', ideal_err)

# increase rank

rank += 1

X0 = np.zeros((N, rank))
X0[:, :-1] = left_orthogonal_base[0]

Y0 = np.zeros((rank, M))
Y0[:-1,:] = left_orthogonal_base[1]

QX, RX = np.linalg.qr(X0, mode='reduced')
Y1 = RX @ Y0

UY, ssY, VtY = np.linalg.svd(Y1, full_matrices=False)

ssY[-1] = ssY[-2] / 3
Y2 = UY @ np.diag(ssY) @ VtY

base = (QX, Y2)
left_orthogonal_base = left_orthogonalize_base(base)

#

J, (relerrs, relerrs_r) = objective(left_orthogonal_base, inputs, true_outputs)
g0, _ = gradient_func(left_orthogonal_base, inputs, true_outputs)
g = standardize_perturbation(left_orthogonal_base, g0)
norm_g = dumb_norm(g)

print('J=', J)
print('relerrs=', relerrs)
print('relerrs_r=', relerrs_r)
print('||g||=', norm_g)

for _ in range(30):
    apply_H = lambda P: gn_hessian_matvec(left_orthogonal_base, P, inputs)

    M_helper = make_inner_product_helper_matrix(left_orthogonal_base)
    ip_func = lambda p1, p2: inner_product_of_tangent_vectors(
        standardize_perturbation(left_orthogonal_base, p1), standardize_perturbation(left_orthogonal_base, p2), M_helper
    )

    trust_radius = 1e0
    rtol = 0.1

    p, aux = cg_steihaug(
        apply_H, g, add_tangent_vectors, scale_tangent_vector, ip_func, trust_radius, rtol, display=True,
    )

    base = retract_tangent_vector(left_orthogonal_base, p)
    left_orthogonal_base = left_orthogonalize_base(base)

    J, (relerrs, relerrs_r) = objective(left_orthogonal_base, inputs, true_outputs)
    g0, _ = gradient_func(left_orthogonal_base, inputs, true_outputs)
    g = standardize_perturbation(left_orthogonal_base, g0)
    norm_g = dumb_norm(g)

    print('J=', J)
    print('relerrs=', np.sqrt(relerrs))
    print('relerrs_r=', np.sqrt(relerrs_r))
    print('||g||=', norm_g)

A2 = base_to_full(left_orthogonal_base)
computed_err = np.linalg.norm(A2 - A) / np.linalg.norm(A)
print('rank=', rank)
print('computed_err=', computed_err)

U, ss, Vt = np.linalg.svd(A)
Ar = U[:,:rank] @ np.diag(ss[:rank]) @ Vt[:rank,:]

ideal_err = np.linalg.norm(Ar - A) / np.linalg.norm(A)
print('ideal_err=', ideal_err)

# increase rank

rank += 1

X0 = np.zeros((N, rank))
X0[:, :-1] = left_orthogonal_base[0]

Y0 = np.zeros((rank, M))
Y0[:-1,:] = left_orthogonal_base[1]

QX, RX = np.linalg.qr(X0, mode='reduced')
Y1 = RX @ Y0

UY, ssY, VtY = np.linalg.svd(Y1, full_matrices=False)

ssY[-1] = ssY[-2] / 3
Y2 = UY @ np.diag(ssY) @ VtY

base = (QX, Y2)
left_orthogonal_base = left_orthogonalize_base(base)

#

J, (relerrs, relerrs_r) = objective(left_orthogonal_base, inputs, true_outputs)
g0, _ = gradient_func(left_orthogonal_base, inputs, true_outputs)
g = standardize_perturbation(left_orthogonal_base, g0)
norm_g = dumb_norm(g)

print('J=', J)
print('relerrs=', relerrs)
print('relerrs_r=', relerrs_r)
print('||g||=', norm_g)

for _ in range(30):
    apply_H = lambda P: gn_hessian_matvec(left_orthogonal_base, P, inputs)

    M_helper = make_inner_product_helper_matrix(left_orthogonal_base)
    ip_func = lambda p1, p2: inner_product_of_tangent_vectors(
        standardize_perturbation(left_orthogonal_base, p1), standardize_perturbation(left_orthogonal_base, p2), M_helper
    )

    trust_radius = 1e0
    rtol = 0.1

    p, aux = cg_steihaug(
        apply_H, g, add_tangent_vectors, scale_tangent_vector, ip_func, trust_radius, rtol, display=True,
    )

    base = retract_tangent_vector(left_orthogonal_base, p)
    left_orthogonal_base = left_orthogonalize_base(base)

    J, (relerrs, relerrs_r) = objective(left_orthogonal_base, inputs, true_outputs)
    g0, _ = gradient_func(left_orthogonal_base, inputs, true_outputs)
    g = standardize_perturbation(left_orthogonal_base, g0)
    norm_g = dumb_norm(g)

    print('J=', J)
    print('relerrs=', np.sqrt(relerrs))
    print('relerrs_r=', np.sqrt(relerrs_r))
    print('||g||=', norm_g)

A2 = base_to_full(left_orthogonal_base)
computed_err = np.linalg.norm(A2 - A) / np.linalg.norm(A)
print('rank=', rank)
print('computed_err=', computed_err)

U, ss, Vt = np.linalg.svd(A)
Ar = U[:,:rank] @ np.diag(ss[:rank]) @ Vt[:rank,:]

ideal_err = np.linalg.norm(Ar - A) / np.linalg.norm(A)
print('ideal_err=', ideal_err)















#### trust region optimization

def _update_trust_region_size(
        J_old,          # scalar. E.g., float or array with shape=()
        J_true,         # scalar
        J_predicted,    # scalar
        old_trust_region_radius, # scalar
        min_trust_region_radius, # scalar
        max_trust_region_radius, # scalar
        termination_reason: str,
        display:            bool = True,
) -> typ.Tuple[
    typ.Any, # scalar. new_trust_region_radius
    typ.Any, # scalar. reduction_ratio
]:
    def _printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    J_old = np.array(J_old) # increase precision; (x-y)/(x-z) has big rounding errors when x, y, z are all close
    J_true = np.array(J_true)
    J_predicted = np.array(J_predicted)
    actual_drop = J_old - J_true
    predicted_drop = J_old - J_predicted
    reduction_ratio = actual_drop / predicted_drop
    relative_reduction = actual_drop / J_true
    s = '{:<15s}'.format('Jd_old') + '{:<15s}'.format('Jd_true') + '{:<15s}'.format('Jd_predicted') + '\n'
    s += '{:<15.5E}'.format(J_old) + '{:<15.5E}'.format(J_true) + '{:<15.5E}'.format(J_predicted) + '\n'
    _printmaybe(s)
    _printmaybe('Drop in Jd from previous Newton iteration:')
    s = '{:<12s}'.format('relative') + '{:<12s}'.format('predicted') + '{:<12s}'.format(
        'actual') + '{:<18s}'.format('predicted/actual') + '\n'
    s += '{:<12.1E}'.format(relative_reduction) + '{:<12.1E}'.format(predicted_drop) + '{:<12.1E}'.format(
        actual_drop) + '{:<18.1E}'.format(reduction_ratio) + '\n'
    _printmaybe(s)

    # Modification of Algorithm 4.1 in Nocedal and Wright, page 69.
    # if reduction_ratio < 0.25:
    if (0.75 * J_old < J_true - 0.25 * J_predicted) or (J_true < 0.0) or (J_predicted < 0.0): # Mathematically equivalent but more numerically stable
        new_trust_region_radius = jnp.maximum(0.25 * old_trust_region_radius, min_trust_region_radius)
        s = ('reducing trust region size: '
             + '{:>8.1E}'.format(old_trust_region_radius)
             + ' -> '
             + '{:<8.1E}'.format(new_trust_region_radius)
             + '\n')
        _printmaybe(s)
    else:
        # if reduction_ratio > 0.75 and termination_reason.lower() in {
        #     'exited_trust_region', 'encountered_negative_curvature'
        # }:
        if 0.25 * J_old > J_true - 0.75 * J_predicted and termination_reason.lower() in {
            'exited_trust_region', 'encountered_negative_curvature'
        }:
            new_trust_region_radius = jnp.minimum(2.0 * old_trust_region_radius, max_trust_region_radius)
            s = ('increasing trust region size: ' +
                 '{:>8.1E}'.format(old_trust_region_radius) +
                 ' -> ' +
                 '{:<8.1E}'.format(new_trust_region_radius) +
                 '\n')
            _printmaybe(s)
        else:
            new_trust_region_radius = old_trust_region_radius
            s = ('keeping same trust region size: '
                 + '{:>8.1E}'.format(new_trust_region_radius)
                 )
            _printmaybe(s)

    return new_trust_region_radius, reduction_ratio

#

# M_helper = make_inner_product_helper_matrix(left_orthogonal_base)
# sqrtM_helper = spd_sqrtm(M_helper)
# isqrtM_helper = jnp.linalg.inv(sqrtM_helper)
#
# MSp = apply_tangent_mass_matrix(standardize_perturbation(left_orthogonal_base, perturbation), isqrtM_helper)
# SMp = standardize_perturbation(apply_tangent_mass_matrix(left_orthogonal_base, isqrtM_helper), perturbation)
#
# print(np.linalg.norm(MSp[0] - SMp[0]))
# print(np.linalg.norm(MSp[1] - SMp[1]))