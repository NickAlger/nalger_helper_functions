import numpy as np
import jax.numpy as jnp
import jax
import typing as typ
import functools as ft
import scipy.linalg as sla
import matplotlib.pyplot as plt

from jax.experimental import sparse

from nalger_helper_functions.experimental.low_rank_matrix_optimization.low_rank_matrix_manifold import *
from nalger_helper_functions.rsvd import rsvd_double_pass
from nalger_helper_functions.experimental.low_rank_matrix_optimization.low_rank_matrix_optimization_problem import *
from nalger_helper_functions.experimental.low_rank_matrix_optimization.low_rank_matrix_optimizers import *
import nalger_helper_functions.tree_linalg as tla
from nalger_helper_functions import laplacian_nd, mass_matrix_nd

# jax.config.update("jax_enable_x64", True) # enable double precision
# from jax.config import config
# config.update('jax_disable_jit', True)


#### Use trust region method to solve low rank fit problem

def forward_map(
        m: typ.Tuple[
            jnp.ndarray, # X
            jnp.ndarray, # Y
        ], # A = (X @ Y)**2 (elementwise square)
        x: typ.Tuple[
            jnp.ndarray, # Omega
            jnp.ndarray, # Omega_r
        ],
) -> typ.Tuple[
    jnp.ndarray, # Z = A @ Omega
    jnp.ndarray, # Z_r = Omega_r @ A
]:
    X, Y = m
    Omega, Omega_r = x

    A = X @ Y # slow. just for testing
    B = A**2 # jnp.cos(A) * A

    Y = B @ Omega
    Y_r = Omega_r @ B
    return Y, Y_r


def loss_func(
        Z: typ.Tuple[
            jnp.ndarray, # Y
            jnp.ndarray, # Y_r
        ],
        Ztrue: typ.Tuple[
            jnp.ndarray, # Ytrue
            jnp.ndarray, # Ytrue_r
        ],
) ->  typ.Tuple[
    jnp.ndarray,  # Jd, shape=()
    typ.Tuple[
        jnp.ndarray,  # relerrs
        jnp.ndarray,  # relerrs_r
    ],
]:
    y, y_r = Z
    ytrue, ytrue_r = Ztrue

    rsq_num = jnp.sum((y - ytrue)**2, axis=0) ** 2
    rsq_den = jnp.sum(ytrue**2, axis=0) ** 2
    rsq_num_r = jnp.sum((y_r - ytrue_r)**2, axis=1) ** 2
    rsq_den_r = jnp.sum(ytrue_r**2, axis=1) ** 2
    relerrs = rsq_num / rsq_den
    relerrs_r = rsq_num_r / rsq_den_r

    # Jd = 0.5 * jnp.sum(rsq_num) + 0.5 * jnp.sum(rsq_num_r)
    Jd = 0.5 * jnp.sum(relerrs) + 0.5 * jnp.sum(relerrs_r)
    # Jd = 0.5 * jnp.sum(rsq_num) / py.size + 0.5 * jnp.sum(rsq_num_r) / py_r.size
    return Jd, (relerrs, relerrs_r)


def regularization0_func(
        m: typ.Tuple[
            jnp.ndarray,  # X
            jnp.ndarray,  # Y
        ],  # A = X**2 @ cos(Y)
        a_reg,
) -> jnp.ndarray: # scalar, shape=()
    X, Y = m
    return a_reg * 0.5 * jnp.sum((X.T @ X) * (Y @ Y.T))


#

N = 100
M = 89
true_rank = 50
num_samples = 10

U, _, Vt = np.linalg.svd(np.random.randn(N, M), full_matrices=False)
ss = 1.0 / np.arange(1, np.minimum(N, M)+1)**2
A = U @ np.diag(ss) @ Vt
B = A**2 #jnp.cos(A) * A

inputs = (jnp.array(np.random.randn(M, num_samples)), jnp.array(np.random.randn(num_samples, N)))
true_outputs = (B @ inputs[0], inputs[1] @ B)

#

rank = 5
num_samples = true_outputs[0].shape[1]
noise_level = 5e-2

m0 = (U[:, :rank], np.diag(ss[:rank]) @ Vt[:rank, :])
noise = tla.randn(m0)
noise = tla.scale(tla.scale(noise, tla.div(tla.norm(m0), tla.norm(noise))), noise_level)
m0 = tla.add(m0, noise)

# m0 = (U[:,:rank], ss[0]*Vt[:rank,:])

# m0 = (jnp.array(np.random.randn(N, rank)), jnp.array(np.random.randn(rank, M)))
# m0 = svd_initial_guess(true_outputs, rank)
# m0 = (jnp.sqrt(jnp.abs(m0[0])), jnp.sqrt(jnp.abs(m0[1])))

aa = [1e-8]
# aa = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
errs = []
for a_reg in aa:
    m0 = left_orthogonalize_low_rank(m0)
    m, previous_step = low_rank_manifold_trust_region_optimize_fixed_rank_nonlinear(
        inputs, true_outputs, m0, forward_map, loss_func, lambda m: regularization0_func(m, a_reg),
        newton_max_iter=50, newton_rtol=1e-2,
        cg_rtol_power=0.5,
        # cg_rtol_power=1.0,
    )

    A2 = low_rank_to_full(m)
    B2 = A2**2 # np.cos(A2) * A2
    computed_err = np.linalg.norm(B2 - B) / np.linalg.norm(B)
    print('rank=', rank)
    print('computed_err=', computed_err)
    errs.append(computed_err)

    U, ss, Vt = np.linalg.svd(A)
    Ar = U[:, :rank] @ np.diag(ss[:rank]) @ Vt[:rank, :]
    Br = Ar**2

    ideal_err = np.linalg.norm(Br - B) / np.linalg.norm(B)
    print('ideal_err=', ideal_err)

    Ursvd, ssrsvd, Vtrsvd = rsvd_double_pass(
        (N, M), lambda X: A @ X, lambda Z: Z @ A, rank, num_samples-rank,
    )

    Arsvd = Ursvd @ np.diag(ssrsvd) @ Vtrsvd
    Brsvd = Arsvd**2

    rsvd_err = np.linalg.norm(Brsvd - B) / np.linalg.norm(B)
    print('rsvd_err=', rsvd_err)

    svals = np.linalg.svd(m[1])[1]
    print('svals=', svals)

    m = m0

print('aa=', aa)
print('errs=', errs)

#
#
# delta_rank = 1
#
# delta_outputs = tla.sub(true_outputs, forward_map(m0, inputs))
# dm0 = svd_initial_guess(delta_outputs, delta_rank)
#
# dm, previous_step = low_rank_manifold_trust_region_optimize_fixed_rank_nonlinear(
#     inputs,
#     delta_outputs,
#     dm0,
#     forward_map,
#     loss_func,
#     lambda m: regularization0_func(m, a_reg),
#     newton_max_iter=50,
#     newton_rtol=1e-2,
#     cg_rtol_power=0.5,
#     # cg_rtol_power=1.0,
# )
#
# m = add_low_rank_matrices([m, dm])
#
# rank = m[0].shape[1]
#
# A2 = low_rank_to_full(m)
# computed_err = np.linalg.norm(A2**2 - A_squared_elementwise) / np.linalg.norm(A_squared_elementwise)
# print('rank=', rank)
# print('computed_err=', computed_err)
#
# U, ss, Vt = np.linalg.svd(A)
# Ar = U[:, :rank] @ np.diag(ss[:rank]) @ Vt[:rank, :]
#
# ideal_err = np.linalg.norm(Ar - A) / np.linalg.norm(A)
# print('ideal_err=', ideal_err)
#
# Ursvd, ssrsvd, Vtrsvd = rsvd_double_pass(
#     (N, M), lambda X: A @ X, lambda Z: Z @ A, rank, num_samples-rank,
# )
#
# Arsvd = Ursvd @ np.diag(ssrsvd) @ Vtrsvd
#
# rsvd_err = np.linalg.norm(Arsvd - A) / np.linalg.norm(A)
# print('rsvd_err=', rsvd_err)
#
# svals = np.linalg.svd(m[1])[1]
# print('svals=', svals)
#
# m0 = left_orthogonalize_low_rank(m)


