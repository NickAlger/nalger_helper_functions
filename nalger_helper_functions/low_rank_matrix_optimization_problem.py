import numpy as np
import jax.numpy as jnp
import jax
import typing as typ
import functools as ft

import nalger_helper_functions.tree_linalg as tla


__all__ = [
    'forward_map',
    'misfit',
    'misfit_gradient',
    'forward_map_jvp',
    'forward_map_vjp',
    'misfit_gn_hessian_matvec',
    'regularization',
    'regularization_gradient',
    'regularization_hessian_matvec',
    'objective',
    'gradient',
    'gn_hessian_vector_product',
    'spd_sqrtm',
    'tangent_space_misfit',
]


# jax.config.update("jax_enable_x64", True) # enable double precision

OptVar          = typ.TypeVar('OptVar')
OptVarTangent   = typ.TypeVar('OptVarTangent')
OptVarCoTangent = typ.TypeVar('OptVarCoTangent')

Inputs = typ.TypeVar('Inputs')

Outputs             = typ.TypeVar('Outputs')
OutputsTangent      = typ.TypeVar('OutputsTangent')
OutputsCoTangent    = typ.TypeVar('OutputsCoTangent')

Obs             = typ.TypeVar('Obs')
ObsTangent      = typ.TypeVar('ObsTangent')
ObsCoTangent    = typ.TypeVar('ObsCoTangent')

Misfits = typ.TypeVar('Misfits')
MisfitsTangent = typ.TypeVar('MisfitsTangent')
MisfitsCoTangent = typ.TypeVar('MisfitsCoTangent')

ScalarType  = typ.TypeVar('ScalarType')


default_obs_operator        = lambda u: u
default_subtract_obs        = tla.tree_sub
default_residual_to_misfits = tla.tree_normsquared_leaves
default_combine_misfits     = lambda m: 0.5 * tla.tree_sum(m)

default_combine_misfits_gradient    = lambda m: 0.5 * tla.tree_ones(m)
default_residual_to_misfits_vjp     = lambda dmisfits, residual: tla.tree_mult(tla.tree_scale(dmisfits, 2.0), residual)
default_obs_operator_transpose      = lambda u: u

def misfit(
        x:                  OptVar,
        inputs:             Inputs,
        true_observations:  Obs,
        forward_map:            typ.Callable[[OptVar, Inputs],  Outputs],
        obs_operator:           typ.Callable[[Outputs],         Obs]        = default_obs_operator,
        subtract_obs:           typ.Callable[[Obs, Obs],        Obs]        = default_subtract_obs,
        residual_to_misfits:    typ.Callable[[Obs],             Misfits]    = default_residual_to_misfits,
        combine_misfits:        typ.Callable[[Misfits],         ScalarType] = default_combine_misfits,
) -> typ.Tuple[
    ScalarType, # J, combined misfit
    typ.Tuple[
        Obs,     # residual
        Misfits, # all misfits
    ],
]:
    observations    = obs_operator(forward_map(x, inputs))
    residual        = subtract_obs(true_observations, observations)
    misfits         = residual_to_misfits(residual)
    J               = combine_misfits(misfits)
    return J, (residual, misfits)


def misfit_gradient(
        x:          OptVar,
        inputs:     Inputs,
        misfits:    Misfits,
        residual:   Obs,
        forward_map_vjp:            typ.Callable[[OutputsCoTangent, OptVar, Inputs],    OptVarCoTangent],
        combine_misfits_gradient:   typ.Callable[[Misfits],                             MisfitsCoTangent]   = default_combine_misfits_gradient,
        residual_to_misfits_vjp:    typ.Callable[[MisfitsCoTangent, Obs],               ObsCoTangent]       = default_residual_to_misfits_vjp,
        obs_operator_transpose:     typ.Callable[[ObsCoTangent],                        OutputsCoTangent]   = default_obs_operator_transpose,
) -> OptVarCoTangent:
    return forward_map_vjp(
        obs_operator_transpose(
            residual_to_misfits_vjp(
                combine_misfits_gradient(misfits), residual)
        ), x, inputs)


def misfit_gauss_newton_hessian_matvec(
        x: OptVarType,
        p: OptTangentType,
        inputs: InputType,
        observation_operator: typ.Callable[[typ.Tuple[jnp.ndarray, jnp.ndarray]], typ.Any], # returns linalg tree with same shape as true_outputs
        observation_operator_transpose: typ.Callable[[typ.Any], typ.Tuple[jnp.ndarray, jnp.ndarray]], # takes in linalg tree with same shape as true_outputs
): # p -> J^T B^T B J p
    return forward_map_vjp(
        base, inputs,
        observation_operator_transpose(
            observation_operator(
                forward_map_jvp(base, perturbation, inputs)
            )
        )
    )



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


def regularization(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        apply_ML: typ.Callable[[jnp.ndarray], jnp.ndarray], # X -> ML @ X
        apply_MR: typ.Callable[[jnp.ndarray], jnp.ndarray], # Y -> Y @ MR
): # 1/2 * ||ML @ X @ Y @ MR||^2
    X, Y = base
    Xhat = apply_ML(X)
    Yhat = apply_MR(Y)
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


def objective(
        left_orthogonal_base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        inputs: typ.Tuple[
            jnp.ndarray,  # Omega, shape=(M,k)
            jnp.ndarray,  # Omega_r, shape=(k_r,N)
        ],
        true_outputs: typ.Tuple[
            jnp.ndarray,  # Ztrue, shape=(N',k)
            jnp.ndarray,  # Ztrue_r, shape=(k_r,M')
        ],
        left_observation_operator:              typ.Callable[[jnp.ndarray], jnp.ndarray],  # (N,   k)  -> (N',  k)
        right_observation_operator:             typ.Callable[[jnp.ndarray], jnp.ndarray],  # (k_r, M)  -> (k_r, M')
        a_reg,
        apply_ML: typ.Callable[[jnp.ndarray], jnp.ndarray],  # X -> ML @ X
        apply_MR: typ.Callable[[jnp.ndarray], jnp.ndarray],  # Y -> Y @ MR
):
    Jd, (residual, residual_norms_squared) = misfit(
        left_orthogonal_base, inputs, true_outputs, left_observation_operator, right_observation_operator
    )
    Jr0 = regularization(left_orthogonal_base, apply_ML, apply_MR)
    Jr = a_reg * Jr0
    J = Jd + Jr
    return J, (Jd, Jr, residual, residual_norms_squared)


def gradient(
        left_orthogonal_base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        inputs: typ.Tuple[
            jnp.ndarray,  # Omega, shape=(M,k)
            jnp.ndarray,  # Omega_r, shape=(k_r,N)
        ],
        residual: typ.Tuple[
            jnp.ndarray,  # forward_residual. shape=(N',k)
            jnp.ndarray,  # reverse_residual. shape=(k_r, M')
        ],
        left_observation_operator_transpose: typ.Callable[[jnp.ndarray], jnp.ndarray],  # (N',  k)  -> (N,   k)
        right_observation_operator_transpose: typ.Callable[[jnp.ndarray], jnp.ndarray],  # (k_r, M') -> (k_r, M)
        a_reg,
        apply_ML: typ.Callable[[jnp.ndarray], jnp.ndarray],  # X -> ML @ X
        apply_MLT: typ.Callable[[jnp.ndarray], jnp.ndarray],
        apply_MR: typ.Callable[[jnp.ndarray], jnp.ndarray],  # Y -> Y @ MR
        apply_MRT: typ.Callable[[jnp.ndarray], jnp.ndarray],
):
    gd = misfit_gradient(
        left_orthogonal_base, inputs, residual,
        left_observation_operator_transpose, right_observation_operator_transpose
    )
    gr0 = regularization_gradient(left_orthogonal_base, apply_ML, apply_MLT, apply_MR, apply_MRT)
    gr = tla.tree_scale(gr0, a_reg)
    g = tla.tree_add(gd, gr)
    return g, (gd, gr)


def gn_hessian_vector_product(
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
        left_observation_operator: typ.Callable[[jnp.ndarray], jnp.ndarray],  # (N,   k)  -> (N',  k)
        right_observation_operator: typ.Callable[[jnp.ndarray], jnp.ndarray],  # (k_r, M)  -> (k_r, M')
        left_observation_operator_transpose: typ.Callable[[jnp.ndarray], jnp.ndarray],  # (N',  k)  -> (N,   k)
        right_observation_operator_transpose: typ.Callable[[jnp.ndarray], jnp.ndarray],  # (k_r, M') -> (k_r, M)
        a_reg,
        apply_ML: typ.Callable[[jnp.ndarray], jnp.ndarray],  # X -> ML @ X
        apply_MLT: typ.Callable[[jnp.ndarray], jnp.ndarray],
        apply_MR: typ.Callable[[jnp.ndarray], jnp.ndarray],  # Y -> Y @ MR
        apply_MRT: typ.Callable[[jnp.ndarray], jnp.ndarray],
):
    Hd_p = misfit_gn_hessian_matvec(left_orthogonal_base, perturbation, inputs)
    Hr_p0 = regularization_hessian_matvec(left_orthogonal_base, perturbation, apply_ML, apply_MLT, apply_MR, apply_MRT)
    Hr_p = tla.tree_scale(Hr_p0, a_reg)
    Hp = tla.tree_add(Hd_p, Hr_p)
    return Hp


@jax.jit
def spd_sqrtm(A):
    ee, P = jnp.linalg.eigh(A)
    return P @ (jnp.sqrt(jnp.abs(ee)).reshape((-1,1)) * P.T)


@jax.jit
def tangent_space_misfit(
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
