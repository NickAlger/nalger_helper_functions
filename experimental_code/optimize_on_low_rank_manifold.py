import numpy as np
import jax.numpy as jnp
import jax
import typing as typ
import functools as ft

from nalger_helper_functions.low_rank_matrix_manifold import *
from nalger_helper_functions import cg_steihaug, trust_region_optimize
import nalger_helper_functions.tree_linalg as tla

import matplotlib.pyplot as plt

# jax.config.update("jax_enable_x64", True) # enable double precision

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
def misfit(
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
    Y, Y_r = forward_map(base, inputs) # predicted outputs
    rsq = jnp.sum((Y - Ytrue)**2, axis=0) # k numbers
    rsq_r = jnp.sum((Y_r - Ytrue_r)**2, axis=1)
    J = 0.5 * jnp.sum(rsq) + 0.5 * jnp.sum(rsq_r)
    return J, (rsq, rsq_r)





gradient_func = jax.jit(jax.grad(misfit, argnums=0, has_aux=True))


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
    '''Returns:
    Z, Z_r = lim s->0 forward_map((X + s*dX, Y + s*dY), Omega) - forward_map((X, Y), Omega) / s
    '''
    X, Y = base
    dX, dY = perturbation
    # dX, dY = tangent_oblique_projection(base, perturbation)

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
    '''<forward_map_jvp(x,i,p), z> = <p, forward_map_vjp(x,i,z)>
    < . , .> is dumb frobenius norm
    '''
    X, Y = base
    Z, Z_r = ZZ

    Omega, Omega_r = inputs
    dX = jnp.einsum('ix,aj,jx->ia', Z, Y, Omega) + jnp.einsum('xi,aj,xj->ia', Omega_r, Y, Z_r)
    dY = jnp.einsum('ix,ia,jx->aj', Z, X, Omega) + jnp.einsum('xi,ia,xj->aj', Omega_r, X, Z_r)

    return dX, dY # <-- agrees with vjp autodiff
    # return tangent_oblique_projection_transpose(base, (dX, dY)) # <-- agrees with vjp autodiff

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
): # p -> J^T J p
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
    J0, _ = misfit(left_orthogonal_base, inputs, true_outputs)

    p = perturbation

    g, _ = gradient_func(left_orthogonal_base, inputs, true_outputs)

    gp = tla.tree_dot(g, p)

    Hp = gn_hessian_matvec(left_orthogonal_base, p, inputs)

    pHp = tla.tree_dot(p, Hp)

    return 0.5 * pHp + gp + J0


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

J, (rsq, rsq_r) = misfit(base, inputs, true_outputs)

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

Q, ss, Vt = jnp.linalg.svd(X, full_matrices=False)
R = ss.reshape((-1, 1)) * Vt

# Q, R = jnp.linalg.qr(X, mode='reduced') # <-- more efficient but less stable?

left_orthogonal_base = left_orthogonalize_low_rank(base)
standard_perturbation = tangent_oblique_projection(left_orthogonal_base, perturbation)

Z = np.random.randn(*Ytrue.shape)
Z_r = np.random.randn(*Ytrue_r.shape)
ZZ = (Z, Z_r)

Jp = forward_map_jvp(left_orthogonal_base, standard_perturbation, inputs)
JtZ = forward_map_vjp(left_orthogonal_base, inputs, ZZ)

t1 = tla.tree_dot(Jp, ZZ) # np.sum(Jp[0] * ZZ[0]) + np.sum(Jp[1] * ZZ[1])
t2 = tla.tree_dot(JtZ, standard_perturbation) # np.sum(JtZ[0] * standard_perturbation[0]) + np.sum(JtZ[1] + standard_perturbation[1])

err_forward_map_vjp = np.abs(t1 - t2) / np.abs(t1 + t2)
print('err_forward_map_vjp=', err_forward_map_vjp)

#

J = tangent_space_objective(
    left_orthogonal_base, standard_perturbation, inputs, true_outputs
)

big_base = attached_tangent_vector_as_low_rank(left_orthogonal_base, standard_perturbation)
J_true, _ = misfit(big_base, inputs, true_outputs)

err_tangent_space_objective = np.abs(J - J_true) / np.abs(J_true)
print('err_tangent_space_objective=', err_tangent_space_objective)


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

# not using this
def sls_iter(
        x0: typ.Tuple[
            jnp.ndarray, # shape=(N,r)
            jnp.ndarray, # shape=(r,M)
        ],
        inputs: typ.Tuple[
            jnp.ndarray, # Omega, shape=(M,k)
            jnp.ndarray, # Omega_r, shape=(k_r,N)
        ],
        true_outputs: typ.Tuple[
            jnp.ndarray, # Ztrue, shape=(N,k)
            jnp.ndarray, # Ztrue_r, shape=(k_r, M)
        ],
) -> typ.Tuple[
    jnp.ndarray, # shape=(N,r)
    jnp.ndarray, # shape=(r,M)
]: # x1
    Omega, Omega_r = inputs
    Ztrue, Ztrue_r = true_outputs

    X0, Y0 = right_orthogonalize_low_rank(x0)

    X1 = Ztrue @ np.linalg.pinv(Y0 @ Omega)

    X1b, Y1b = left_orthogonalize_low_rank((X1, Y0))

    Y2 = np.linalg.pinv(Omega_r @ X1b) @ Ztrue_r
    x2 = (X1b, Y2)
    return x2


# not using this
def alscg(
        x0: typ.Tuple[
            jnp.ndarray, # shape=(N,r) # X
            jnp.ndarray, # shape=(r,M) # Y
        ],
        compute_x_aux: typ.Callable,                # x                         -> x_aux
        full_objective_func: typ.Callable,          # x, x_aux                  -> (J, J_aux)
        full_gradient_func: typ.Callable,           # x, x_aux, J_aux           -> (gX, gY), g_aux
        full_gn_hessian_matvec_func: typ.Callable,  # x, (pX, pY), x_aux, g_aux -> (HpX, HpY)
        num_iter: int,
        cg_rtol: float,
        cg_max_iter: int = 25,
) -> typ.Tuple[
    jnp.ndarray, # shape=(N,r)
    jnp.ndarray, # shape=(r,M)
]: # x1
    x = x0

    for ii in range(num_iter):
        x = right_orthogonalize_low_rank(x)

        x_aux = compute_x_aux(x)
        J, J_aux = full_objective_func(x, x_aux)
        g, g_aux = full_gradient_func(x, x_aux, J_aux)
        gX, gY = g
        norm_gX = jnp.sqrt(jnp.sum(gX)**2)
        norm_gY = jnp.sqrt(jnp.sum(gY)**2)

        print('iter X:', ii)
        print('J=', J)
        print('J_aux=', J_aux)
        print('norm_gX=', norm_gX)
        print('norm_gY=', norm_gY)

        gn_hessian_matvec_X = lambda pX: full_gn_hessian_matvec_func(x, (pX, 0.0*x[1]), x_aux, J_aux, g_aux)[0]
        pX, info = cg_steihaug(gn_hessian_matvec_X, gX, lambda u, v,: u + v, lambda u, c: c * u,
                               lambda u, v: jnp.sum(u * v), np.inf, cg_rtol, max_iter=cg_max_iter, display=True)

        x = (x[0] + pX, x[1])

        x = left_orthogonalize_low_rank(x)

        x_aux = compute_x_aux(x)
        J, J_aux = full_objective_func(x, x_aux)
        g, g_aux = full_gradient_func(x, x_aux, J_aux)
        gX, gY = g
        norm_gX = jnp.sqrt(jnp.sum(gX)**2)
        norm_gY = jnp.sqrt(jnp.sum(gY)**2)

        print('iter Y:', ii)
        print('J=', J)
        print('J_aux=', J_aux)
        print('norm_gX=', norm_gX)
        print('norm_gY=', norm_gY)

        alscgY_hessian_matvec = lambda pY: full_gn_hessian_matvec_func(x, (0.0*x[0], pY), x_aux, J_aux, g_aux)[1]

        pY, info = cg_steihaug(alscgY_hessian_matvec, gY, lambda u, v,: u + v, lambda u, c: c * u,
                               lambda u, v: jnp.sum(u * v), np.inf, cg_rtol, max_iter=cg_max_iter, display=True)

        x = (x[0], x[1] + pY)

    x_aux = compute_x_aux(x)
    J, J_aux = full_objective_func(x, x_aux)
    g, g_aux = full_gradient_func(x, x_aux, J_aux)
    gX, gY = g
    norm_gX = jnp.sqrt(jnp.sum(gX) ** 2)
    norm_gY = jnp.sqrt(jnp.sum(gY) ** 2)

    print('final iter:')
    print('J=', J)
    print('J_aux=', J_aux)
    print('norm_gX=', norm_gX)
    print('norm_gY=', norm_gY)

    return x


# not using this
def change_rank(
        previous_step,
        new_rank,
        small_singular_value_parameter = 0.5,
):
    x_prev, p_prev, x_aux_prev = previous_step

    (X0, Y0) = retract_arbitrary_rank(x_prev, p_prev, x_aux_prev, new_rank)

    Q, R = np.linalg.qr(X0, mode='reduced')

    U0, ss0, Vt0 = np.linalg.svd(R @ Y0, full_matrices=False)
    U = U0[:,:new_rank]
    Vt = Vt0[:new_rank,:]

    old_rank = len(ss0)
    ss = np.zeros(new_rank)
    ss[:old_rank] = ss0[:old_rank]
    ss[old_rank:] = ss0[old_rank-1] * small_singular_value_parameter
    X2 = Q @ U

    Y2 = np.diag(ss) @ Vt

    new_x = (X2, Y2)
    return new_x

#

@jax.jit
def compute_x_aux(x):
    M_helper = make_tangent_mass_matrix_helper(x)
    sqrtM_helper = spd_sqrtm(M_helper)
    isqrtM_helper = jnp.linalg.inv(sqrtM_helper)
    iM_helper = jnp.linalg.inv(M_helper)
    return M_helper, sqrtM_helper, isqrtM_helper, iM_helper


J_func = jax.jit(lambda x, x_aux: misfit(
    x,                      # arguments used by optimizer
    inputs, true_outputs,   # arguments removed by partial application
))

def manifold_gradent(
        # arguments used by optimizer:
        x, x_aux, J_aux,
        # arguments removed by partial application:
        flat_gradient, # x, inputs, true_outputs, J_aux -> g_flat, g_aux
        inputs, true_outputs,
):
    M_helper, sqrtM_helper, isqrtM_helper, iM_helper = x_aux
    g0, g_aux = flat_gradient(x, inputs, true_outputs, J_aux)
    # g1 = apply_tangent_mass_matrix(g0, isqrtM_helper) # <-- correct
    # g1 = apply_tangent_mass_matrix(g0, sqrtM_helper)
    g1 = g0 # <-- Correct if mass matrix used as cg_steihaug preconditioner
    g2 = tangent_orthogonal_projection(x, g1)
    return g2, g_aux

def manifold_hessian_matvec(
        # arguments used by optimizer
        x, p, x_aux, J_aux, g_aux,
        # arguments removed by partial application
        flat_hessian_matvec, # x, p, inputs, J_aux, g_aux -> H_flat(x) @ p
        inputs,
):
    M_helper, sqrtM_helper, isqrtM_helper, iM_helper = x_aux
    p2 = tangent_orthogonal_projection(x, p)
    # p3 = apply_tangent_mass_matrix(p2, isqrtM_helper) # <-- Correct if mass matrix used as cg_steihaug preconditioner
    p3 = p2
    Hp0 = flat_hessian_matvec(x, p3, inputs, J_aux, g_aux)
    # Hp1 = apply_tangent_mass_matrix(Hp0, isqrtM_helper) # <-- correct
    # Hp1 = apply_tangent_mass_matrix(Hp0, sqrtM_helper)
    Hp1 = Hp0 # <-- Correct if mass matrix used as cg_steihaug preconditioner
    Hp2 = tangent_orthogonal_projection(x, Hp1)
    return Hp2


def retract_arbitrary_rank(
        x, p, x_aux, rank,
):
    M_helper, sqrtM_helper, isqrtM_helper, iM_helper = x_aux
    p2 = tangent_orthogonal_projection(x, p)
    # p3 = apply_tangent_mass_matrix(p2, isqrtM_helper) # <-- correct
    # p3 = apply_tangent_mass_matrix(p2, sqrtM_helper)
    p3 = p2
    x_plus_p = retract_tangent_vector(x, p3, rank)
    return x_plus_p

retract = lambda x, p, x_aux: retract_arbitrary_rank(x, p, x_aux, None)

# add     = lambda u, v, x, x_aux: add_sequences(u, v)
# scale   = lambda u, c, x, x_aux: scale_sequence(u, c)
# dual_pairing = lambda u, v, x, x_aux: inner_product_of_sequences(u, v)
preconditioner_apply = lambda u, x, x_aux, J_aux, g_aux: apply_tangent_mass_matrix(u, x_aux[0])
preconditioner_solve = lambda u, x, x_aux, J_aux, g_aux: apply_tangent_mass_matrix(u, x_aux[3])

# J_aux_callback = lambda J_aux: print(str(J_aux[0]) + '\n' + str(J_aux[1]))

def J_aux_callback(J_aux):
    relerrs, relerrs_r = J_aux
    s = '\nRelative errors forward:\n'
    for ii in range(len(relerrs)):
        s += "{:<10.2e}".format(relerrs[ii])
    s += '\nRelative errors reverse:\n'
    for ii in range(len(relerrs_r)):
        s += "{:<10.2e}".format(relerrs_r[ii])
    s += '\n'
    print(s)


def low_rank_manifold_trust_region_optimize_fixed_rank(
        inputs,
        true_outputs,
        x0,
        **kwargs,
):
    g_func = jax.jit(lambda x, x_aux, J_aux: manifold_gradent(
        x, x_aux, J_aux,
        lambda x, i, t_o, J_aux: gradient_func(x, i, t_o), # flat_gradient
        inputs, true_outputs,
    ))
    H_matvec_func = jax.jit(lambda p, x, x_aux, J_aux, g_aux: manifold_hessian_matvec(
        x, p, x_aux, J_aux, g_aux,
        lambda x, p, i, J_aux, g_aux: gn_hessian_matvec(x, p, i), # flat_hessian_matvec
        inputs,
    ))
    return trust_region_optimize(
        J_func,
        g_func,
        H_matvec_func,
        x0,
        retract=retract,
        preconditioner_apply=preconditioner_apply,
        preconditioner_solve=preconditioner_solve,
        compute_x_aux=compute_x_aux,
        J_aux_callback=J_aux_callback,
        **kwargs,
    )


def svd_initial_guess(
        true_outputs,
        rank,
):
    Z, Z_r = true_outputs
    X0 = np.linalg.svd(Z)[0][:, :rank]
    Y0 = np.linalg.svd(Z_r)[2][:rank, :]
    x0 = (X0, Y0)
    return left_orthogonalize_low_rank(x0)

#

def rsvd(
        A_matvecs: typ.Callable,  # X -> A X, A has shape (N,M), X has shape (M, k1)
        A_rmatvecs: typ.Callable, # Z -> Z A, A has shape (N,M), Z has shape (k2, N)
        r: int, # rank
        p: int, # oversampling parameter
) -> typ.Tuple[
    np.ndarray, # U, shape=(N,r)
    np.ndarray, # ss, shape=(r,)
    np.ndarray, # Vt, shape=(r,M)
]:
    Omega = np.random.randn(M,r+p)
    Y = A_matvecs(Omega)
    Q, R = np.linalg.qr(Y)
    B = A_rmatvecs(Q.T)
    U0, ss0, Vt0 = np.linalg.svd(B, full_matrices=False)
    U = Q @ U0[:,:r]
    ss = ss0[:r]
    Vt = Vt0[:r,:]
    return U, ss, Vt

#

rank = 5
num_samples = true_outputs[0].shape[1]

x0 = svd_initial_guess(true_outputs, rank)

x = x0
# x = sls_iter(x0, inputs, true_outputs)

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

Ursvd, ssrsvd, Vtrsvd = rsvd(
    lambda X: A @ X, lambda Z: Z @ A, rank, num_samples-rank,
)

Arsvd = Ursvd @ np.diag(ssrsvd) @ Vtrsvd

rsvd_err = np.linalg.norm(Arsvd - A) / np.linalg.norm(A)
print('rsvd_err=', rsvd_err)

svals = np.linalg.svd(x[1])[1]
print('svals=', svals)

# rank = rank + 1
# x0 = change_rank(previous_step, rank, small_singular_value_parameter = 0.5) # <-- converges to saddle  point


if False:
    for ii in range(4):
        outputs = forward_map(x, inputs)

        delta_outputs = (true_outputs[0] - outputs[0], true_outputs[1] - outputs[1])

        delta_x0 = left_orthogonalize_low_rank(svd_initial_guess(delta_outputs, 1))

        J_before_delta, relerr_before_delta = J_func(delta_x0, None)

        delta_x, delta_previous_step = low_rank_manifold_trust_region_optimize_fixed_rank(
            inputs, delta_outputs, delta_x0,
            newton_max_steps=1, cg_rtol_power=0.5, newton_rtol=0.5, cg_max_iter=1,
        )

        x = add_low_rank_matrices([x, delta_x])

        rank = x[0].shape[1]

        A2 = low_rank_to_full(x)
        computed_err = np.linalg.norm(A2 - A) / np.linalg.norm(A)
        print('rank=', rank)
        print('computed_err=', computed_err)

        U, ss, Vt = np.linalg.svd(A)
        Ar = U[:, :rank] @ np.diag(ss[:rank]) @ Vt[:rank, :]

        ideal_err = np.linalg.norm(Ar - A) / np.linalg.norm(A)
        print('ideal_err=', ideal_err)

        svals = np.linalg.svd(x[1])[1]
        print('svals=', svals)

        #

        x0 = left_orthogonalize_low_rank(x)

        J_before, relerr_before = J_func(x0, None)


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

        svals = np.linalg.svd(x[1])[1]
        print('svals=', svals)


