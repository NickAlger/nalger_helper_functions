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
    'misfit_gauss_newton_hessian_matvec',
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

Param          = typ.TypeVar('Param')
ParamTangent   = typ.TypeVar('ParamTangent')
ParamCoTangent = typ.TypeVar('ParamCoTangent')

Inputs = typ.TypeVar('Inputs')

State           = typ.TypeVar('State')
StateTangent    = typ.TypeVar('StateTangent')
StateCoTangent  = typ.TypeVar('StateCoTangent')

ForwardAux      = typ.TypeVar('ForwardAux')
ForwardJvpAux   = typ.TypeVar('ForwardJvpAux')
ForwardVjpAux   = typ.TypeVar('ForwardVjpAux')

Outputs             = typ.TypeVar('Outputs')
OutputsTangent      = typ.TypeVar('OutputsTangent')
OutputsCoTangent    = typ.TypeVar('OutputsCoTangent')

LossAux = typ.TypeVar('LossAux')
Scalar  = typ.TypeVar('Scalar')



default_observe = lambda u: u
default_observe_jvp = lambda u, du: du
default_observe_vjp = lambda u, dy: dy

def default_compute_loss(y: Outputs, y_true: Outputs) -> typ.Tuple[Scalar, LossAux]:
    all_num_squared = tla.tree_normsquared_leaves(tla.tree_sub(y_true, y))
    all_den_squared = tla.tree_normsquared_leaves(y_true)
    all_relative_squared_errors = tla.tree_div(all_num_squared, all_den_squared)
    J = 0.5 * tla.tree_sum(all_num_squared)
    return J, all_relative_squared_errors

def default_compute_loss_jvp(y: Outputs, y_true: Outputs, dy: OutputsTangent) -> Scalar:
    return tla.tree_dot(y, dy)

def default_compute_loss_vjp(y: Outputs, y_true: Outputs, dJ: Scalar) -> OutputsCoTangent:
    return tla.tree_scale(y, dJ)


default_subtract_obs        = tla.tree_sub
default_residual_to_misfits = tla.tree_normsquared_leaves
default_combine_misfits     = lambda m: 0.5 * tla.tree_sum(m)

default_combine_misfits_gradient    = lambda misfits: 0.5 * tla.tree_ones(misfits)
default_residual_to_misfits_vjp     = lambda residual, misfits_cotangent: tla.tree_mult(tla.tree_scale(misfits_cotangent, 2.0), residual)
default_obs_operator_vjp            = lambda outputs, obs_cotangent: obs_cotangent


def misfit(
        m:              Param,
        x:              Inputs,
        y_true:         Outputs, # true observations
        forward_map:    typ.Callable[[Param, Inputs],       typ.Tuple[State,    ForwardAux]],
        observe:        typ.Callable[[State],               Outputs]                            = default_observe,
        compute_loss:   typ.Callable[[Outputs, Outputs],    typ.Tuple[Scalar,   LossAux]]   = default_compute_loss, # true_obs, obs -> J, J_aux
) -> typ.Tuple[
    Scalar, # misfit Jd
    LossAux
]:
    u, state_aux    = forward_map(m, x)
    y               = observe(u)
    J, J_aux        = compute_loss(y, y_true)
    return J, (J_aux, y, u, state_aux)


def misfit_gradient(
        m:      Param,
        x:      Inputs,
        u:      State,
        f_aux:  ForwardAux,
        y:      Outputs,
        y_true: Outputs,
        forward_map_vjp:    typ.Callable[[Param, Inputs, StateCoTangent, ForwardAux],   typ.Tuple[ParamCoTangent, ForwardVjpAux]],
        observe_vjp:        typ.Callable[[State, OutputsCoTangent],                     StateCoTangent]   = default_observe_vjp,
        compute_loss_vjp:   typ.Callable[[Outputs, Outputs, Scalar],                    OutputsCoTangent] = default_compute_loss_vjp,
) -> ParamCoTangent:
    z1: OutputsCoTangent    = compute_loss_vjp(y, y_true, 1.0)
    z2: StateCoTangent      = observe_vjp(u, z1)
    g, fvjp_aux = forward_map_vjp(m, z2, x, f_aux)
    return g, fvjp_aux



def misfit_gauss_newton_hessian_matvec(
        dm: ParamTangent,
        m:  Param,
        x:  Inputs,
        u:  State,
        f_aux: ForwardAux,
        y: Outputs,
        y_true: Outputs,
        forward_map_jvp:    typ.Callable[[Param, Inputs, ParamTangent, ForwardAux],   typ.Tuple[StateTangent, ForwardJvpAux]],
        forward_map_vjp:    typ.Callable[[Param, Inputs, StateCoTangent, ForwardAux], typ.Tuple[ParamCoTangent, ForwardVjpAux]],
        observe_jvp:        typ.Callable[[State, StateTangent],     OutputsTangent] = default_observe_jvp,
        observe_vjp:        typ.Callable[[State, OutputsCoTangent], StateCoTangent] = default_observe_vjp,
        compute_loss_jvp:   typ.Callable[[Outputs, Outputs, OutputsTangent],    Scalar]              = default_compute_loss_jvp,
        compute_loss_vjp:   typ.Callable[[Outputs, Outputs, Scalar],            OutputsCoTangent]    = default_compute_loss_vjp,
) -> ParamCoTangent:
    du, fjvp_aux = forward_map_jvp(m, dm, x, f_aux)
    dy = observe_jvp(x, du)
    dJ = compute_loss_jvp(y, y_true, dy)
    dy2 = compute_loss_vjp(y, y_true, dJ)
    du2 = observe_vjp(x, dy2)
    Hdm, fvjp_aux = forward_map_vjp(m, )
    return Hdm, (fjvp_aux, fvjp_aux)




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
