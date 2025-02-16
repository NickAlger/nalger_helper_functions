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



if False:
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

        (X0, Y0) = retract_tangent_vector(x_prev, p_prev, x_aux_prev, new_rank)

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





