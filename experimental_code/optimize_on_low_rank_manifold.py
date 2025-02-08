import numpy as np
import jax.numpy as jnp
import jax
import typing as typ

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
def make_inner_product_helper_matrix(
        left_orthogonal_base: typ.Tuple[
            jnp.ndarray,  # Q, shape=(N,r), Q^T Q = I
            jnp.ndarray,  # Y, shape=(r,M)
        ],
) -> jnp.ndarray: # inner_product_helper_matrix=YY^T, shape=(r,r)
    return Y @ Y.T


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
    dX1_perp, dY1 = standard_perturbation1
    dX2_perp, dY2 = standard_perturbation2
    t1 = jnp.einsum('ia,ab,ib', dX1_perp, inner_product_helper_matrix,  dX2_perp)
    t2 = np.sum(dY1 * dY2)
    return t1 + t2


@jax.jit
def normsquared_of_tangent_vector(
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
    return inner_product_of_tangent_vectors(standard_perturbation1, standard_perturbation2, inner_product_helper_matrix)


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
