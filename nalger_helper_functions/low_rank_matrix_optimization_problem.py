import numpy as np
import jax.numpy as jnp
import jax
import typing as typ
import functools as ft

import nalger_helper_functions.tree_linalg as tla
import nalger_helper_functions.least_squares_framework as lsf


__all__ = [
    'forward_map',
    'forward_map_jvp',
    'forward_map_vjp',
    'regularization',
    'regularization_gradient',
    'regularization_hessian_matvec',
    'spd_sqrtm',
    'objective',
    'gradient',
    'gauss_newton_hessian_matvec',
    # 'tangent_space_misfit',
]


# jax.config.update("jax_enable_x64", True) # enable double precision


# regularization_gauss_newton_hessian_matvec: typ.Callable[[Param, ParamTangent], ParamCoTangent] = default_regularization_gauss_newton_hessian_matvec,

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
def forward_map_jvp(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        inputs: typ.Tuple[
            jnp.ndarray, # Omega, shape=(M,k)
            jnp.ndarray, # Omega_r, shape=(k_r,N)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
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


def regularization(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        apply_R: typ.Callable[[typ.Tuple[jnp.ndarray, jnp.ndarray]], typ.Tuple[jnp.ndarray, jnp.ndarray]],
): # 1/2 * ||ML @ X @ Y @ MR||^2
    Xhat, Yhat = apply_R(base)
    return 0.5 * np.sum((Xhat.T @ Xhat) * (Yhat @ Yhat.T))


def regularization_gradient(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        apply_R: typ.Callable[[typ.Tuple[jnp.ndarray, jnp.ndarray]], typ.Tuple[jnp.ndarray, jnp.ndarray]],
        apply_RT: typ.Callable[[typ.Tuple[jnp.ndarray, jnp.ndarray]], typ.Tuple[jnp.ndarray, jnp.ndarray]],
):
    X, Y = base
    Xhat, Yhat = apply_R(base)
    return apply_RT((Xhat @ (Yhat @ Yhat.T), (Xhat.T @ Xhat) @ Yhat))


def regularization_hessian_matvec(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
        apply_R: typ.Callable[[typ.Tuple[jnp.ndarray, jnp.ndarray]], typ.Tuple[jnp.ndarray, jnp.ndarray]],
        apply_RT: typ.Callable[[typ.Tuple[jnp.ndarray, jnp.ndarray]], typ.Tuple[jnp.ndarray, jnp.ndarray]],
):
    Xhat, Yhat = apply_R(base)
    PXhat, PYhat = apply_R(perturbation)
    PYY = PYhat @ Yhat.T
    PXX = PXhat.T @ Xhat
    HX, HY = apply_RT((PXhat @ (Yhat @ Yhat.T) + Xhat @ (PYY + PYY.T), (PXX + PXX.T) @ Yhat + (Xhat.T @ Xhat) @ PYhat))
    return HX, HY

#

def loss(Y, Y_true):
    y, y_r = Y
    yt, yt_r = Y_true
    rsq_num = np.linalg.norm(y - yt, axis=0) ** 2
    rsq_den = np.linalg.norm(yt, axis=0) ** 2
    rsq_num_r = np.linalg.norm(y_r - yt_r, axis=1) ** 2
    rsq_den_r = np.linalg.norm(yt_r, axis=1) ** 2
    relerrs = rsq_num / rsq_den
    relerrs_r = rsq_num_r / rsq_den_r

    J = 0.5 * np.sum(rsq_num) + 0.5 * np.sum(rsq_num_r)
    return J, (relerrs, relerrs_r)


def objective(
        base, inputs, true_outputs,
        a_reg=0.0, apply_R=lambda u: u,
):
    J, (Jd, Jr, _, state, outputs, Jd_aux) = lsf.objective(
        base, inputs, true_outputs,
        lambda b, i: (forward_map(b, i), None),
        a_reg,
        observe=lambda u: u,
        regularization_function=lambda b: regularization(b, apply_R),
        loss=loss,
    )
    return J, (Jd, Jr, state, outputs, Jd_aux)


def gradient(
        base, inputs, J_aux, true_outputs,
        a_reg=0.0, apply_R=lambda u: u, apply_RT=lambda u: u,
):
    _, _, state, outputs, _ = J_aux
    g, (gd, gr, _) = lsf.gradient(
        base, inputs, state, None, outputs, true_outputs, a_reg,
        lambda b, i, z, f_aux: (forward_map_vjp(b, i, z), None),
        observe_vjp=lambda u, du: du,
        regularization_gradient=lambda b: regularization_gradient(b, apply_R, apply_RT)
    )
    return g, (gd, gr)


def gauss_newton_hessian_matvec(
    perturbation, base, inputs, J_aux,
        a_reg=0.0, apply_R=lambda u: u, apply_RT=lambda u:u,
):
    _, _, state, outputs, _ = J_aux
    H_dm, (Hd_dm, Hr_dm, fjvp_aux, fvjp_aux) = lsf.gauss_newton_hessian_matvec(
        perturbation, base, inputs, state, None, outputs,
        lambda b, i, p, f_aux: (forward_map_jvp(b, i, p), None),
        lambda b, i, z, f_aux: (forward_map_vjp(b, i, z), None),
        a_reg,
        observe_jvp=lambda u, du: du,
        observe_vjp=lambda u, du: du,
        regularization_gnhvp=lambda b, p: regularization_hessian_matvec(b, p, apply_R, apply_RT,),
    )
    return H_dm

#
# def objective(
#         left_orthogonal_base: typ.Tuple[
#             jnp.ndarray,  # X, shape=(N,r)
#             jnp.ndarray,  # Y, shape=(r,M)
#         ],
#         inputs: typ.Tuple[
#             jnp.ndarray,  # Omega, shape=(M,k)
#             jnp.ndarray,  # Omega_r, shape=(k_r,N)
#         ],
#         true_outputs: typ.Tuple[
#             jnp.ndarray,  # Ztrue, shape=(N',k)
#             jnp.ndarray,  # Ztrue_r, shape=(k_r,M')
#         ],
#         left_observation_operator:              typ.Callable[[jnp.ndarray], jnp.ndarray],  # (N,   k)  -> (N',  k)
#         right_observation_operator:             typ.Callable[[jnp.ndarray], jnp.ndarray],  # (k_r, M)  -> (k_r, M')
#         a_reg,
#         apply_ML: typ.Callable[[jnp.ndarray], jnp.ndarray],  # X -> ML @ X
#         apply_MR: typ.Callable[[jnp.ndarray], jnp.ndarray],  # Y -> Y @ MR
# ):
#     Jd, (residual, residual_norms_squared) = misfit(
#         left_orthogonal_base, inputs, true_outputs, left_observation_operator, right_observation_operator
#     )
#     Jr0 = regularization(left_orthogonal_base, apply_ML, apply_MR)
#     Jr = a_reg * Jr0
#     J = Jd + Jr
#     return J, (Jd, Jr, residual, residual_norms_squared)
#
#
# def gradient(
#         left_orthogonal_base: typ.Tuple[
#             jnp.ndarray,  # X, shape=(N,r)
#             jnp.ndarray,  # Y, shape=(r,M)
#         ],
#         inputs: typ.Tuple[
#             jnp.ndarray,  # Omega, shape=(M,k)
#             jnp.ndarray,  # Omega_r, shape=(k_r,N)
#         ],
#         residual: typ.Tuple[
#             jnp.ndarray,  # forward_residual. shape=(N',k)
#             jnp.ndarray,  # reverse_residual. shape=(k_r, M')
#         ],
#         left_observation_operator_transpose: typ.Callable[[jnp.ndarray], jnp.ndarray],  # (N',  k)  -> (N,   k)
#         right_observation_operator_transpose: typ.Callable[[jnp.ndarray], jnp.ndarray],  # (k_r, M') -> (k_r, M)
#         a_reg,
#         apply_ML: typ.Callable[[jnp.ndarray], jnp.ndarray],  # X -> ML @ X
#         apply_MLT: typ.Callable[[jnp.ndarray], jnp.ndarray],
#         apply_MR: typ.Callable[[jnp.ndarray], jnp.ndarray],  # Y -> Y @ MR
#         apply_MRT: typ.Callable[[jnp.ndarray], jnp.ndarray],
# ):
#     gd = misfit_gradient(
#         left_orthogonal_base, inputs, residual,
#         left_observation_operator_transpose, right_observation_operator_transpose
#     )
#     gr0 = regularization_gradient(left_orthogonal_base, apply_ML, apply_MLT, apply_MR, apply_MRT)
#     gr = tla.tree_scale(gr0, a_reg)
#     g = tla.tree_add(gd, gr)
#     return g, (gd, gr)
#
#
# def gn_hessian_vector_product(
#         left_orthogonal_base: typ.Tuple[
#             jnp.ndarray,  # X, shape=(N,r)
#             jnp.ndarray,  # Y, shape=(r,M)
#         ],
#         perturbation: typ.Tuple[
#             jnp.ndarray,  # dX, shape=(N,r)
#             jnp.ndarray,  # dY, shape=(r,M)
#         ],
#         inputs: typ.Tuple[
#             jnp.ndarray,  # Omega, shape=(M,k)
#             jnp.ndarray,  # Omega_r, shape=(k_r,N)
#         ],
#         left_observation_operator: typ.Callable[[jnp.ndarray], jnp.ndarray],  # (N,   k)  -> (N',  k)
#         right_observation_operator: typ.Callable[[jnp.ndarray], jnp.ndarray],  # (k_r, M)  -> (k_r, M')
#         left_observation_operator_transpose: typ.Callable[[jnp.ndarray], jnp.ndarray],  # (N',  k)  -> (N,   k)
#         right_observation_operator_transpose: typ.Callable[[jnp.ndarray], jnp.ndarray],  # (k_r, M') -> (k_r, M)
#         a_reg,
#         apply_ML: typ.Callable[[jnp.ndarray], jnp.ndarray],  # X -> ML @ X
#         apply_MLT: typ.Callable[[jnp.ndarray], jnp.ndarray],
#         apply_MR: typ.Callable[[jnp.ndarray], jnp.ndarray],  # Y -> Y @ MR
#         apply_MRT: typ.Callable[[jnp.ndarray], jnp.ndarray],
# ):
#     Hd_p = misfit_gn_hessian_matvec(left_orthogonal_base, perturbation, inputs)
#     Hr_p0 = regularization_hessian_matvec(left_orthogonal_base, perturbation, apply_ML, apply_MLT, apply_MR, apply_MRT)
#     Hr_p = tla.tree_scale(Hr_p0, a_reg)
#     Hp = tla.tree_add(Hd_p, Hr_p)
#     return Hp


@jax.jit
def spd_sqrtm(A):
    ee, P = jnp.linalg.eigh(A)
    return P @ (jnp.sqrt(jnp.abs(ee)).reshape((-1,1)) * P.T)


# @jax.jit
# def tangent_space_misfit(
#         left_orthogonal_base: typ.Tuple[
#             jnp.ndarray,  # X, shape=(N,r)
#             jnp.ndarray,  # Y, shape=(r,M)
#         ],
#         perturbation: typ.Tuple[
#             jnp.ndarray,  # dX, shape=(N,r)
#             jnp.ndarray,  # dY, shape=(r,M)
#         ],
#         inputs: typ.Tuple[
#             jnp.ndarray,  # Omega, shape=(M,k)
#             jnp.ndarray,  # Omega_r, shape=(k_r,N)
#         ],
#         true_outputs: typ.Tuple[
#             jnp.ndarray,  # Ztrue, shape=(N,k)
#             jnp.ndarray,  # Ztrue_r, shape=(k_r,M)
#         ],
# ):
#     J0, _ = misfit(left_orthogonal_base, inputs, true_outputs)
#     p = perturbation
#     g, _ = misfit_gradient_func(left_orthogonal_base, inputs, true_outputs)
#     gp = tla.tree_dot(g, p)
#     Hp = misfit_gn_hessian_matvec(left_orthogonal_base, p, inputs)
#     pHp = tla.tree_dot(p, Hp)
#     return 0.5 * pHp + gp + J0
