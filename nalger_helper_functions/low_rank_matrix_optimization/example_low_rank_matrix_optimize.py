import numpy as np
import jax.numpy as jnp
import jax
import typing as typ
import functools as ft
import scipy.linalg as sla
import matplotlib.pyplot as plt

from nalger_helper_functions.low_rank_matrix_optimization.low_rank_matrix_manifold import *
from nalger_helper_functions.rsvd import rsvd_double_pass
from nalger_helper_functions.low_rank_matrix_optimization.low_rank_matrix_optimization_problem import *
from nalger_helper_functions.low_rank_matrix_optimization.low_rank_matrix_optimizers import *
import nalger_helper_functions.tree_linalg as tla

# jax.config.update("jax_enable_x64", True) # enable double precision


#### Use trust region method to solve low rank fit problem

N = 100
M = 89
num_samples = 10

U, _, Vt = np.linalg.svd(np.random.randn(N, M), full_matrices=False)
# ss = np.logspace(-30, 0, np.minimum(N,M))
ss = 1.0 / np.arange(1, np.minimum(N, M)+1)**2
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
    inputs, true_outputs, x0,
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
noise_level = 2e-1
num_samples = 10

def laplacian(n):
    return np.diag(2*np.ones(n), 0) + np.diag(-np.ones(n-1), -1) + np.diag(-np.ones(n-1), 1)

ML = 3e-2 * np.eye(N) + laplacian(N)
CL = np.linalg.inv(ML)
MR = 3e-2 * np.eye(M) + laplacian(M)
CR = np.linalg.inv(MR)

K = np.minimum(N,M)

A0 = (CL @ np.random.randn(N,K)) @ (np.random.randn(K,M) @ CR)

noise = np.random.randn(*A0.shape)
noise = noise * noise_level * np.linalg.norm(A0) / np.linalg.norm(noise)
A = A0 + noise

Omega = jnp.array(np.random.randn(M, num_samples))
Omega_r = jnp.array(np.random.randn(num_samples, N))

Omega = Omega / jnp.linalg.norm(Omega, axis=0).reshape((1,-1))
Omega_r = Omega_r / jnp.linalg.norm(Omega_r, axis=1).reshape((-1,1))

Ytrue = A @ Omega
Ytrue_r = Omega_r @ A
inputs = (Omega, Omega_r)
true_outputs = (Ytrue, Ytrue_r)


rank = 5

a_reg = 1e0

RL = a_reg * laplacian(N)
RR = a_reg * laplacian(M)
# RL = a_reg * ML
# RR = a_reg * MR
# RL = ML
# RR = MR

iRL = np.linalg.inv(RL)
iRR = np.linalg.inv(RR)

apply_R = lambda b: (RL @ b[0], b[1] @ RR)
solve_R = lambda b: (iRL @ b[0], b[1] @ iRR)

U, ss, Vt = np.linalg.svd(0.5 * (Omega_r @ Ytrue + Ytrue_r @ Omega), full_matrices=False)
Ax_X = Ytrue @ Vt[:rank,:].T @ np.diag(1.0 / ss[:rank])
Ax_Y = U[:,:rank].T @ Ytrue_r
Ax = Ax_X @ Ax_Y
# Ax_smooth = low_rank_to_full(solve_R((Ax_X, Ax_Y)))

x0 = (Ax_X, Ax_Y)
# x0 = solve_R((Ax_X, Ax_Y))
# x0 = svd_initial_guess(solve_R(true_outputs), rank)
# x0 = svd_initial_guess(true_outputs, rank)
# x0 = solve_R((np.random.randn(N,rank), np.random.randn(rank, M)))
x0 = left_orthogonalize_low_rank(x0)

x, previous_step = low_rank_manifold_trust_region_optimize_fixed_rank(
    inputs, true_outputs, x0,
    apply_R=apply_R,
    newton_max_iter=50, newton_rtol=1e-2,
    cg_rtol_power=0.5,
    # cg_rtol_power=1.0,
)

# inputs_hat = (iRR @ Omega, Omega_r @ iRL)
# true_outputs_hat = (RL @ Ytrue, Ytrue_r @ RR)
# x0_hat = (RL @ x0[0], x0[1] @ RR)
# x0 = left_orthogonalize_low_rank(x0)
#
# xhat, previous_step = low_rank_manifold_trust_region_optimize_fixed_rank(
#     inputs_hat, true_outputs_hat, x0_hat,
#     apply_R = lambda b: (b[0], b[1]),
#     apply_P = lambda b: (iRL @ b[0], b[1] @ iRR),
#     newton_max_iter=50, newton_rtol=1e-2,
#     cg_rtol_power=0.5,
#     # cg_rtol_power=1.0,
# )
# x = (iRL @ xhat[0], xhat[1] @ iRR)

A2 = low_rank_to_full(x)
optimization_err = np.linalg.norm(A2 - A0) / np.linalg.norm(A0)
print('rank=', rank)
print('optimization_err=', optimization_err)

U, ss, Vt = np.linalg.svd(A)
Ar = U[:, :rank] @ np.diag(ss[:rank]) @ Vt[:rank, :]

direct_svd_err = np.linalg.norm(Ar - A0) / np.linalg.norm(A0)
print('direct_svd_err=', direct_svd_err)

num_samples = true_outputs[0].shape[1]
Ursvd, ssrsvd, Vtrsvd = rsvd_double_pass(
    (N, M), lambda X: A @ X, lambda Z: Z @ A, rank, num_samples-rank,
)

single_rsvd_err = np.linalg.norm(Ax - A0) / np.linalg.norm(A0)
print('single_rsvd_err=', single_rsvd_err)

Arsvd = Ursvd @ np.diag(ssrsvd) @ Vtrsvd

double_rsvd_err = np.linalg.norm(Arsvd - A0) / np.linalg.norm(A0)
print('double_rsvd_err=', double_rsvd_err)

svals = np.linalg.svd(x[1])[1]
print('svals=', svals)

#

plt.figure(figsize=(12,8))
plt.subplot(2,3,1)
plt.imshow(A0)
plt.title('Original A')
plt.subplot(2,3,2)
plt.imshow(A)
plt.title('Noisy A')
plt.subplot(2,3,3)
plt.imshow(A2)
plt.title('Optimization')
plt.subplot(2,3,4)
plt.imshow(Ax)
plt.title('One pass rsvd of noisy A')
plt.subplot(2,3,5)
plt.imshow(Ar)
plt.title('Exact svd of noisy A')
plt.subplot(2,3,6)
plt.imshow(Arsvd)
plt.title('Double pass rsvd of noisy A')
#

if False:
    delta_rank = 1

    outputs = forward_map(x0, inputs)
    target_outputs = tla.sub(true_outputs, outputs)

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



