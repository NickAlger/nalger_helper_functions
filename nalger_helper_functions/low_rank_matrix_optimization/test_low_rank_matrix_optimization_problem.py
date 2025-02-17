import numpy as np
import jax
import jax.numpy as jnp

from nalger_helper_functions.low_rank_matrix_optimization.low_rank_matrix_optimization_problem import *
from nalger_helper_functions.low_rank_matrix_optimization.low_rank_matrix_manifold import *
import nalger_helper_functions.tree_linalg as tla

# Test

jax.config.update("jax_enable_x64", True) # enable double precision

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

A2 = low_rank_to_full(base)
Y2true = A2 @ Omega
Y2true_r = Omega_r @ A2

err_forward_map = np.linalg.norm(Y2 - Y2true) / np.linalg.norm(Y2true)
err_forward_map_r = np.linalg.norm(Y2_r - Y2true_r) / np.linalg.norm(Y2true_r)

print('err_forward_map=', err_forward_map)
print('err_forward_map_r=', err_forward_map_r)

#

a_reg = np.random.rand()
apply_R = lambda u: u

J, (Jd, Jr, outputs, (rsq, rsq_r)) = objective(base, inputs, true_outputs, a_reg, apply_R)

#

rsq_true = np.linalg.norm(Y2 - Ytrue, axis=0)**2 / np.linalg.norm(Ytrue, axis=0)**2
rsq_true_r = np.linalg.norm(Y2_r - Ytrue_r, axis=1)**2 / np.linalg.norm(Ytrue_r, axis=1)**2
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

df = forward_map_jvp(base, inputs, perturbation)

s = 1e-6
f = forward_map(base, inputs)
f2 = forward_map((base[0]+s*perturbation[0], base[1] + s*perturbation[1]), inputs)
df_diff = ((f2[0] - f[0]) / s, (f2[1] - f[1]) / s)

err_forward_map_jvp0 = np.linalg.norm(df[0] - df_diff[0]) / np.linalg.norm(df_diff[0])
print('s=', s, ', err_forward_map_jvp0=', err_forward_map_jvp0)

err_forward_map_jvp1 = np.linalg.norm(df[1] - df_diff[1]) / np.linalg.norm(df_diff[1])
print('s=', s, ', err_forward_map_jvp1=', err_forward_map_jvp1)

#

Q, ss, Vt = jnp.linalg.svd(X, full_matrices=False)
R = ss.reshape((-1, 1)) * Vt

# Q, R = jnp.linalg.qr(X, mode='reduced') # <-- more efficient but less stable?

left_orthogonal_base = left_orthogonalize_low_rank(base)
standard_perturbation = tangent_oblique_projection(left_orthogonal_base, perturbation)

Z = np.random.randn(*Ytrue.shape)
Z_r = np.random.randn(*Ytrue_r.shape)
ZZ = (Z, Z_r)

Jp = forward_map_jvp(left_orthogonal_base, inputs, standard_perturbation)
JtZ = forward_map_vjp(left_orthogonal_base, inputs, ZZ)

t1 = tla.dot(Jp, ZZ) # np.sum(Jp[0] * ZZ[0]) + np.sum(Jp[1] * ZZ[1])
t2 = tla.dot(JtZ, standard_perturbation) # np.sum(JtZ[0] * standard_perturbation[0]) + np.sum(JtZ[1] + standard_perturbation[1])

err_forward_map_vjp = np.abs(t1 - t2) / np.abs(t1 + t2)
print('err_forward_map_vjp=', err_forward_map_vjp)

#

# J = tangent_space_misfit(
#     left_orthogonal_base, standard_perturbation, inputs, true_outputs
# )
#
# big_base = attached_tangent_vector_as_low_rank(left_orthogonal_base, standard_perturbation)
# J_true, _ = misfit(big_base, inputs, true_outputs)
#
# err_tangent_space_misfit = np.abs(J - J_true) / np.abs(J_true)
# print('err_tangent_space_misfit=', err_tangent_space_misfit)

#

ML = np.random.randn(N,N)
MR = np.random.randn(M,M)

def apply_R(
        base,
):
    X, Y = base
    return ML @ X, Y @ MR

def apply_RT(
        base,
):
    X, Y = base
    return ML.T @ X, Y @ MR.T


R0 = regularization(base, apply_R)
gR0 = regularization_gradient(base, apply_R)

dX = np.random.randn(N, r)
dY = np.random.randn(r, M)
perturbation = (dX, dY)

dG = regularization_hessian_matvec(base, perturbation, apply_R)

dR = tla.dot(gR0, perturbation)

s = 1e-7
base1 = tla.add(base, tla.scale(perturbation, s))
R1 = regularization(base1, apply_R)
gR1 = regularization_gradient(base1, apply_R)

dR_diff = (R1 - R0) / s
err_regularization_gradient = np.abs(dR_diff - dR) / np.abs(dR_diff)
print('s=', s, ', err_regularization_gradient=', err_regularization_gradient)

dG_diff = tla.scale(tla.sub(gR1, gR0), 1.0 / s)
err_regularization_hessian_matvec = tla.norm(tla.sub(dG_diff, dG)) / tla.norm(dG_diff)
print('s=', s, ', err_regularization_hessian_matvec=', err_regularization_hessian_matvec)

#

a_reg = 0.324

J0, (_, _, y0, _) = objective(
    base, inputs, true_outputs,
    a_reg, apply_R,
)
g0, (gd0, gr0) = gradient(
        base, inputs, y0, true_outputs,
        a_reg, apply_R
)

dJ = tla.dot(g0, perturbation)

s = 1e-7
base1 = tla.add(base, tla.scale(perturbation, s))

J1, (_, _, y1, _) = objective(
    base1, inputs, true_outputs,
    a_reg, apply_R,
)

dJ_diff = (J1 - J0) / s

err_gradient = np.abs(dJ - dJ_diff) / np.abs(dJ_diff)
print('s=', s, ', err_gradient=', err_gradient)

# Finite difference check Gauss-Newton Hessian at location with zero residual

J0, (_, _, y0, _) = objective(
    base, inputs, outputs, # <-- true_outputs=outputs here
    a_reg, apply_R,
)
g0, (gd0, gr0) = gradient(
        base, inputs, y0, outputs,
        a_reg, apply_R,
)
dg, (_, _) = gauss_newton_hessian_matvec(
    perturbation, base, inputs,
    a_reg, apply_R,
)

s = 1e-7
base1 = tla.add(base, tla.scale(perturbation, s))

J1, (_, _, y1, _) = objective(
    base1, inputs, outputs,
    a_reg, apply_R,
)
g1, (gd1, gr1) = gradient(
        base1, inputs, y1, outputs,
        a_reg, apply_R,
)
dg_diff = tla.scale(tla.sub(g1, g0), 1.0 / s)

err_gnhess = tla.norm(tla.sub(dg, dg_diff)) / tla.norm(dg_diff)
print('s=', s, ', err_gnhess=', err_gnhess)

