import numpy as np
import jax.numpy as jnp
import jax
import typing as typ
import functools as ft
import scipy.linalg as sla

from nalger_helper_functions.low_rank_matrix_manifold import *
from nalger_helper_functions.rsvd import rsvd_double_pass
from nalger_helper_functions.low_rank_matrix_optimization_problem import *
from nalger_helper_functions.low_rank_matrix_optimizers import *
import nalger_helper_functions.tree_linalg as tla

# jax.config.update("jax_enable_x64", True) # enable double precision


#### Use trust region method to solve low rank fit problem

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

rank = 5
num_samples = true_outputs[0].shape[1]

x0 = svd_initial_guess(true_outputs, rank)
x0 = left_orthogonalize_low_rank(x0)

x, previous_step = low_rank_manifold_trust_region_optimize_fixed_rank(
    inputs, true_outputs, x0, a_reg=1e-2,
    newton_max_iter=50, newton_rtol=1e-5,
    cg_rtol_power=0.5,
    # cg_rtol_power=1.0,
)

A2 = low_rank_to_full(x)
computed_err = np.linalg.norm(A2 - A) / np.linalg.norm(A)
print('rank=', rank)
print('computed_err=', computed_err)

U, ss, Vt = np.linalg.svd(A)
Ar = U[:, :rank] @ np.diag(ss[:rank]) @ Vt[:rank, :]

ideal_err = np.linalg.norm(Ar - A) / np.linalg.norm(A)
print('ideal_err=', ideal_err)

Ursvd, ssrsvd, Vtrsvd = rsvd_double_pass(
    (N, M), lambda X: A @ X, lambda Z: Z @ A, rank, num_samples-rank,
)

Arsvd = Ursvd @ np.diag(ssrsvd) @ Vtrsvd

rsvd_err = np.linalg.norm(Arsvd - A) / np.linalg.norm(A)
print('rsvd_err=', rsvd_err)

svals = np.linalg.svd(x[1])[1]
print('svals=', svals)


#### Create true matrix by drawing rows and columns from a distribution

N = 100
M = 89

U, _, Vt = np.linalg.svd(np.random.randn(N, N), full_matrices=False)
ss = np.logspace(-3, 0, N)
CL = U @ np.diag(ss) @ Vt
ML = Vt.T @ np.diag(1.0 / ss) @ U.T

U, _, Vt = np.linalg.svd(np.random.randn(M, M), full_matrices=False)
ss = np.logspace(-3, 0, M)
CR = U @ np.diag(ss) @ Vt
MR = Vt.T @ np.diag(1.0 / ss) @ U.T

K = np.minimum(N,M)

A = (CL @ np.random.randn(N,K)) @ np.diag(np.logspace(-24, 0, K)) @ (np.random.randn(K,M) @ CR)

Omega = jnp.array(np.random.randn(M, num_samples))
Omega_r = jnp.array(np.random.randn(num_samples, N))
Ytrue = A @ Omega
Ytrue_r = Omega_r @ A
inputs = (Omega, Omega_r)
true_outputs = (Ytrue, Ytrue_r)


rank = 5

x0 = svd_initial_guess(true_outputs, rank)
# x0 = (ML @ np.random.randn(N,rank), np.random.randn(rank, M) @ MR)
x0 = left_orthogonalize_low_rank(x0)

apply_R = lambda b: (ML @ b[0], b[1] @ MR)
apply_RT = lambda b: (ML.T @ b[0], b[1] @ MR.T)


x, previous_step = low_rank_manifold_trust_region_optimize_fixed_rank(
    inputs, true_outputs, x0,
    a_reg=1e-3,
    apply_R=apply_R, apply_RT=apply_RT,
    newton_max_iter=50, newton_rtol=1e-5,
    cg_rtol_power=0.5,
    # cg_rtol_power=1.0,
)

A2 = low_rank_to_full(x)
computed_err = np.linalg.norm(A2 - A) / np.linalg.norm(A)
print('rank=', rank)
print('computed_err=', computed_err)

U, ss, Vt = np.linalg.svd(A)
Ar = U[:, :rank] @ np.diag(ss[:rank]) @ Vt[:rank, :]

ideal_err = np.linalg.norm(Ar - A) / np.linalg.norm(A)
print('ideal_err=', ideal_err)

num_samples = true_outputs[0].shape[1]
Ursvd, ssrsvd, Vtrsvd = rsvd_double_pass(
    (N, M), lambda X: A @ X, lambda Z: Z @ A, rank, num_samples-rank,
)

Arsvd = Ursvd @ np.diag(ssrsvd) @ Vtrsvd

rsvd_err = np.linalg.norm(Arsvd - A) / np.linalg.norm(A)
print('rsvd_err=', rsvd_err)

svals = np.linalg.svd(x[1])[1]
print('svals=', svals)

#

delta_rank = 1

outputs = forward_map(x0, inputs)
target_outputs = tla.tree_sub(true_outputs, outputs)

x0 = svd_initial_guess(true_outputs, delta_rank)

dx, previous_step = low_rank_manifold_trust_region_optimize_fixed_rank(
    inputs, true_outputs, x0,
    # a_reg=1e-2,
    # apply_ML=apply_ML, apply_MLT=apply_MLT, apply_MR=apply_MR, apply_MRT=apply_MRT,
    newton_max_iter=50, newton_rtol=1e-5,
    # cg_rtol_power=0.5,
    cg_rtol_power=1.0,
)

x = add_low_rank_matrices([x, dx])

rank = x[0].shape[1]

A2 = low_rank_to_full(x)
computed_err = np.linalg.norm(A2 - A) / np.linalg.norm(A)
print('rank=', rank)
print('computed_err=', computed_err)

U, ss, Vt = np.linalg.svd(A)
Ar = U[:, :rank] @ np.diag(ss[:rank]) @ Vt[:rank, :]

ideal_err = np.linalg.norm(Ar - A) / np.linalg.norm(A)
print('ideal_err=', ideal_err)

num_samples = true_outputs[0].shape[1]
Ursvd, ssrsvd, Vtrsvd = rsvd_double_pass(
    (N, M), lambda X: A @ X, lambda Z: Z @ A, rank, num_samples-rank,
)

Arsvd = Ursvd @ np.diag(ssrsvd) @ Vtrsvd

rsvd_err = np.linalg.norm(Arsvd - A) / np.linalg.norm(A)
print('rsvd_err=', rsvd_err)

svals = np.linalg.svd(x[1])[1]
print('svals=', svals)

x0 = left_orthogonalize_low_rank(x)



