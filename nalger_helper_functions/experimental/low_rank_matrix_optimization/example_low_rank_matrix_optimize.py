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

N = 200
M = 189
noise_level = 5e-1

ML0_sparse = (1e4 * mass_matrix_nd([N], [1.0]) + laplacian_nd([N], [1.0], ['N'], ['N']))
ML0 = ML0_sparse.toarray()
# ML1 = ML0 @ ML0
ML_sparse = 1e-2 * ML0_sparse
ML_sparse_jax = sparse.BCOO.from_scipy_sparse(ML_sparse)
ML = ML_sparse.toarray()
# ML2 = 1e-2 * ML1
CL = np.linalg.inv(ML)

MR0_sparse = (1e4 * mass_matrix_nd([M], [1.0]) + laplacian_nd([M], [1.0], ['N'], ['N']))
MR0 = MR0_sparse.toarray()
# MR1 = MR0 @ MR0
MR_sparse = 1e-2 * MR0_sparse
MR_sparse_jax = sparse.BCOO.from_scipy_sparse(MR_sparse)
MR = MR_sparse.toarray()
# MR2 = 1e-2 * MR1
CR = np.linalg.inv(MR)

K = np.minimum(N,M)

# X @ Y where cols of X drawn N(0, ML^2), rows of Y are drawn from N(0, MR^2)
A0 = (CL @ np.random.randn(N,K)) @ (np.random.randn(K,M) @ CR)

num_first_eigs = 10

U, ss, Vt = np.linalg.svd(A0, full_matrices=False)
# A0 = U @ np.diag(np.power(ss, 0.5)) @ Vt
first_ss = np.power(ss[:num_first_eigs], 0.75)
last_ss = np.power(ss[num_first_eigs:], 0.25)
last_ss = last_ss * first_ss[-1] / last_ss[0]
new_ss = np.concatenate([first_ss, last_ss])
A0 = U @ np.diag(new_ss) @ Vt
plt.figure()
plt.semilogy(new_ss)

# # tt = np.linspace(-np.pi, np.pi, N)
# row_multiplier = np.exp(-0.5 * tt**2 / 2)
# # row_multiplier = np.sin(tt)
# A0 = row_multiplier.reshape((-1,1)) * A0

noise = np.random.randn(*A0.shape)
noise = noise * noise_level * np.linalg.norm(A0) / np.linalg.norm(noise)
A = A0 + noise


num_samples = 15

row_inds = np.random.permutation(N)[:num_samples]
col_inds = np.random.permutation(M)[:num_samples]

Omega = np.eye(M)[:, col_inds]
Omega_r = np.eye(N)[row_inds, :]

# Omega = jnp.array(np.random.randn(M, num_samples))
# Omega_r = jnp.array(np.random.randn(num_samples, N))

# Omega = Omega / jnp.linalg.norm(Omega, axis=0).reshape((1,-1))
# Omega_r = Omega_r / jnp.linalg.norm(Omega_r, axis=1).reshape((-1,1))

Ytrue = A @ Omega
Ytrue_r = Omega_r @ A
inputs = (Omega, Omega_r)
true_outputs = (Ytrue, Ytrue_r)


rank = 10

a_reg = 1e-3

# RL = a_reg * laplacian(N)
# RR = a_reg * laplacian(M)


RL_sparse_jax = a_reg * ML_sparse_jax
RR_sparse_jax = a_reg * MR_sparse_jax
RL = a_reg * ML_sparse.toarray()
RR = a_reg * MR_sparse.toarray()
# RL = a_reg * ML # <-- Standard
# RR = a_reg * MR

# RL = a_reg * ML2 # <-- extra smoothing
# RR = a_reg * MR2

# RL = ML
# RR = MR

# min 0.5*||Y - Y_true||^2 + a_reg*0.5*||RL @ X @ Y||^2 + a_reg*0.5*||X @ Y @ RR||^2 + a_reg^2*0.5*||RL @ X @ Y @ RR||^2

iRL = np.linalg.inv(RL)
iRR = np.linalg.inv(RR)

# apply_R = lambda b: (RL @ b[0], b[1] @ RR)
apply_R = lambda b: (RL_sparse_jax @ b[0], b[1] @ RR_sparse_jax)
solve_R = lambda b: (iRL @ b[0], b[1] @ iRR)

# Doing CUR VVVV
U, ss, Vt = np.linalg.svd(0.5 * (Omega_r @ Ytrue + Ytrue_r @ Omega), full_matrices=False)
Ax_X = Ytrue @ Vt[:rank,:].T @ np.diag(1.0 / ss[:rank])
Ax_Y = U[:,:rank].T @ Ytrue_r
Ax = Ax_X @ Ax_Y # CUR approx
# Ax_smooth = low_rank_to_full(solve_R((Ax_X, Ax_Y)))

extra_basis = 10

# B, B_r = solve_R((Ytrue, Ytrue_r))
B, B_r = solve_R((np.random.randn(N,num_samples+extra_basis), np.random.randn(num_samples+extra_basis, M)))
# B, B_r = solve_R((np.random.randn(N, rank), np.random.randn(rank, M)))
Q = np.linalg.qr(B, mode='reduced')[0]
Q_r = np.linalg.qr(B_r.T, mode='reduced')[0].T

projected_Ax_X = Q @ Q.T @ Ax_X
projected_Ax_Y = Ax_Y @ Q_r.T @ Q_r
projected_Ax = projected_Ax_X @ projected_Ax_Y
x0 = (projected_Ax_X, projected_Ax_Y)

# x0 = (Ax_X, Ax_Y)
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
#     # cg_rtol_power=0.5,
#     cg_rtol_power=1.0,
# )
# x = (iRL @ xhat[0], xhat[1] @ iRR)

A2 = low_rank_to_full(x)
optimization_err = np.linalg.norm(A2 - A0) / np.linalg.norm(A0)
print('rank=', rank)
print('optimization_err=', optimization_err)

# U, ss, Vt = np.linalg.svd(A0)
U, ss, Vt = np.linalg.svd(A)
Ar = U[:, :rank] @ np.diag(ss[:rank]) @ Vt[:rank, :]

direct_svd_err = np.linalg.norm(Ar - A0) / np.linalg.norm(A0)
print('direct_svd_err=', direct_svd_err)

num_samples = true_outputs[0].shape[1]
Ursvd, ssrsvd, Vtrsvd = rsvd_double_pass(
    (N, M), lambda X: A @ X, lambda Z: Z @ A, rank, num_samples-rank,
)

cur_err = np.linalg.norm(Ax - A0) / np.linalg.norm(A0)
print('cur_err=', cur_err)

projected_cur_err = np.linalg.norm(projected_Ax - A0) / np.linalg.norm(A0)
print('projected_cur_err=', projected_cur_err)

Arsvd = Ursvd @ np.diag(ssrsvd) @ Vtrsvd

double_rsvd_err = np.linalg.norm(Arsvd - A0) / np.linalg.norm(A0)
print('double_rsvd_err=', double_rsvd_err)

svals = np.linalg.svd(x[1])[1]
print('svals=', svals)

#

plt.figure(figsize=(12,8))
plt.subplot(2,4,1)
plt.imshow(A0)
plt.title('Original A')
plt.subplot(2,4,2)
plt.imshow(A)
plt.title('Noisy A')
plt.subplot(2,4,3)
plt.imshow(A2)
plt.title('Optimization')
plt.subplot(2,4,4)
plt.imshow(projected_Ax)
plt.title('Projected CUR')
plt.subplot(2,4,5)
plt.imshow(Ax)
plt.title('CUR of noisy A')
plt.subplot(2,4,6)
plt.imshow(Ar)
plt.title('Exact svd of noisy A')
plt.subplot(2,4,7)
plt.imshow(Arsvd)
plt.title('Double pass rsvd of noisy A')



k1 = 10
k2 = 106
k3 = 164
plt.figure(figsize=(12,12))
plt.subplot(3,3,1)
plt.plot(A0[:,k1])
plt.plot(Ax[:,k1])
plt.title('CUR 1')

plt.subplot(3,3,2)
plt.plot(A0[:,k1])
plt.plot(projected_Ax[:,k1])
plt.title('projected CUR 1')

plt.subplot(3,3,3)
plt.plot(A0[:,k1])
plt.plot(A2[:,k1])
plt.title('Optimization 1')

plt.subplot(3,3,4)
plt.plot(A0[:,k2])
plt.plot(Ax[:,k2])
plt.title('CUR 2')

plt.subplot(3,3,5)
plt.plot(A0[:,k2])
plt.plot(projected_Ax[:,k2])
plt.title('projected CUR 2')

plt.subplot(3,3,6)
plt.plot(A0[:,k2])
plt.plot(A2[:,k2])
plt.title('Optimization 2')

plt.subplot(3,3,7)
plt.plot(A0[k3,:])
plt.plot(Ax[k3,:])
plt.title('CUR 2')

plt.subplot(3,3,8)
plt.plot(A0[k3,:])
plt.plot(projected_Ax[k3,:])
plt.title('projected CUR 2')

plt.subplot(3,3,9)
plt.plot(A0[k3,:])
plt.plot(A2[k3,:])
plt.title('Optimization 2')


#

# a_reg = 1e2
# x0 = x
#
# RL = a_reg * ML
# RR = a_reg * MR
#
# iRL = np.linalg.inv(RL)
# iRR = np.linalg.inv(RR)
#
# apply_R = lambda b: (RL @ b[0], b[1] @ RR)
# solve_R = lambda b: (iRL @ b[0], b[1] @ iRR)

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



