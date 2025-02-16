import numpy as np
import jax.numpy as jnp
import jax
import typing as typ
import functools as ft

import nalger_helper_functions.tree_linalg as tla


__all__ = [
    'forward_map',
    'misfit',
    'misfit_gradient_func',
    'forward_map_jvp',
    'forward_map_vjp',
    'misfit_gn_hessian_matvec',
    'regularization',
    'regularization_gradient',
    'regularization_hessian_matvec',
    'spd_sqrtm',
    'tangent_space_objective',
]


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


misfit_gradient_func = jax.jit(jax.grad(misfit, argnums=0, has_aux=True))


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
def misfit_gn_hessian_matvec(
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


def regularization(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        apply_M_left: typ.Callable[[jnp.ndarray], jnp.ndarray], # X -> ML @ X
        apply_M_right: typ.Callable[[jnp.ndarray], jnp.ndarray], # Y -> Y @ MR
): # 1/2 * ||ML @ X @ Y @ MR||^2
    X, Y = base
    Xhat = apply_M_left(X)
    Yhat = apply_M_right(Y)
    return 0.5 * np.sum((Xhat.T @ Xhat) * (Yhat @ Yhat.T))


def regularization_gradient(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        apply_ML: typ.Callable[[jnp.ndarray], jnp.ndarray],  # X -> ML @ X
        apply_MLT: typ.Callable[[jnp.ndarray], jnp.ndarray],
        apply_MR: typ.Callable[[jnp.ndarray], jnp.ndarray],  # Y -> Y @ MR
        apply_MRT: typ.Callable[[jnp.ndarray], jnp.ndarray],
):
    X, Y = base
    Xhat = apply_ML(X)
    Yhat = apply_MR(Y)
    return apply_MLT(Xhat @ (Yhat @ Yhat.T)), apply_MRT((Xhat.T @ Xhat) @ Yhat)


def regularization_hessian_matvec(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
        apply_ML: typ.Callable[[jnp.ndarray], jnp.ndarray],  # X -> ML @ X
        apply_MLT: typ.Callable[[jnp.ndarray], jnp.ndarray],
        apply_MR: typ.Callable[[jnp.ndarray], jnp.ndarray],  # Y -> Y @ MR
        apply_MRT: typ.Callable[[jnp.ndarray], jnp.ndarray],
):
    X, Y = base
    PX, PY = perturbation
    Xhat = apply_ML(X)
    Yhat = apply_MR(Y)
    PXhat = apply_ML(PX)
    PYhat = apply_MR(PY)
    PYY = PYhat @ Yhat.T
    PXX = PXhat.T @ Xhat
    HX = apply_MLT(PXhat @ (Yhat @ Yhat.T) + Xhat @ (PYY + PYY.T))
    HY = apply_MRT((PXX + PXX.T) @ Yhat + (Xhat.T @ Xhat) @ PYhat)
    return HX, HY


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
    g, _ = misfit_gradient_func(left_orthogonal_base, inputs, true_outputs)
    gp = tla.tree_dot(g, p)
    Hp = misfit_gn_hessian_matvec(left_orthogonal_base, p, inputs)
    pHp = tla.tree_dot(p, Hp)
    return 0.5 * pHp + gp + J0
