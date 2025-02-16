import numpy as np
import jax.numpy as jnp
import jax
import typing as typ
import functools as ft

from nalger_helper_functions.low_rank_matrix_manifold import *
from nalger_helper_functions.rsvd import rsvd_double_pass
from nalger_helper_functions.low_rank_matrix_optimizers import *

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

x = x0
x0 = left_orthogonalize_low_rank(x)

x, previous_step = low_rank_manifold_trust_region_optimize_fixed_rank(
    inputs, true_outputs, x0,
    newton_max_iter=50, cg_rtol_power=0.5, newton_rtol=1e-5,
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

