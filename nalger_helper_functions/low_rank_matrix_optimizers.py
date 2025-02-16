import numpy as np
import jax.numpy as jnp
import jax
import typing as typ
import functools as ft

from nalger_helper_functions.low_rank_matrix_optimization_problem import *
from nalger_helper_functions.low_rank_matrix_manifold import *
from nalger_helper_functions import cg_steihaug, trust_region_optimize
import nalger_helper_functions.tree_linalg as tla
from nalger_helper_functions.rsvd import rsvd_double_pass

# jax.config.update("jax_enable_x64", True) # enable double precision

__all__ = [
    'low_rank_manifold_trust_region_optimize_fixed_rank',
    'svd_initial_guess',
]


def low_rank_manifold_trust_region_optimize_fixed_rank(
        inputs,
        true_outputs,
        x0,
        a_reg = 0.0,
        apply_ML: typ.Callable[[jnp.ndarray], jnp.ndarray] = None,  # X -> ML @ X
        apply_MLT: typ.Callable[[jnp.ndarray], jnp.ndarray] = None,
        apply_MR: typ.Callable[[jnp.ndarray], jnp.ndarray] = None,  # Y -> Y @ MR
        apply_MRT: typ.Callable[[jnp.ndarray], jnp.ndarray] = None,
        solve_ML: typ.Callable[[jnp.ndarray], jnp.ndarray] = None,  # X -> ML^-1 @ X
        solve_MLT: typ.Callable[[jnp.ndarray], jnp.ndarray] = None,
        solve_MR: typ.Callable[[jnp.ndarray], jnp.ndarray] = None,  # Y -> Y @ MR^-1
        solve_MRT: typ.Callable[[jnp.ndarray], jnp.ndarray] = None,
        **kwargs,
):
    apply_ML    = (lambda u: u) if apply_ML     is None else apply_ML
    apply_MLT   = (lambda u: u) if apply_MLT    is None else apply_MLT
    apply_MR    = (lambda u: u) if apply_MR     is None else apply_MR
    apply_MRT   = (lambda u: u) if apply_MRT    is None else apply_MRT

    _J_func         = lambda x, x_aux:                          objective(x, inputs, true_outputs, a_reg, apply_ML, apply_MR)
    _g_func         = lambda x, x_aux,  J_aux:                  projected_gradient(x, inputs, true_outputs, a_reg, apply_ML, apply_MLT, apply_MR, apply_MRT)
    _H_matvec_func  = lambda p, x,      x_aux, J_aux, g_aux:    projected_hessian_matvec(x, p, inputs, a_reg, apply_ML, apply_MLT, apply_MR, apply_MRT)
    _apply_M_func   = lambda p, x,      x_aux, J_aux, g_aux:    preconditioner_apply(p, x, x_aux)
    _solve_M_func   = lambda p, x,      x_aux, J_aux, g_aux:    preconditioner_solve(p, x, x_aux)
    _retract_func   = lambda x, p,      x_aux:                  projected_retract(x, p)

    return trust_region_optimize(
        _J_func,
        _g_func,
        _H_matvec_func,
        x0,
        retract=_retract_func,
        preconditioner_apply=_apply_M_func,
        preconditioner_solve=_solve_M_func,
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


@jax.jit
def compute_x_aux(x):
    M_helper = make_tangent_mass_matrix_helper(x)
    iM_helper = jnp.linalg.inv(M_helper)
    return M_helper, iM_helper


def projected_gradient(
        # arguments used by optimizer:
        x,
        # arguments removed by partial application:
        inputs, true_outputs, a_reg, apply_ML, apply_MLT, apply_MR, apply_MRT,
):
    g0, g_aux = gradient(x, inputs, true_outputs, a_reg, apply_ML, apply_MLT, apply_MR, apply_MRT)
    g1 = tangent_orthogonal_projection(x, g0)
    return g1, g_aux


def projected_hessian_matvec(
        # arguments used by optimizer
        x, p,
        # arguments removed by partial application
        inputs, a_reg, apply_ML, apply_MLT, apply_MR, apply_MRT,
):
    p2 = tangent_orthogonal_projection(x, p)
    Hp0 = gn_hessian_vector_product(x, p2, inputs, a_reg, apply_ML, apply_MLT, apply_MR, apply_MRT)
    Hp1 = tangent_orthogonal_projection(x, Hp0)
    return Hp1


def projected_retract_arbitrary_rank(
        x, p, rank,
):
    p2 = tangent_orthogonal_projection(x, p)
    x_plus_p = retract_tangent_vector(x, p2, rank)
    return x_plus_p

projected_retract = lambda x, p: projected_retract_arbitrary_rank(x, p, None)

preconditioner_apply = lambda u, x, x_aux: apply_tangent_mass_matrix(u, x_aux[0])
preconditioner_solve = lambda u, x, x_aux: apply_tangent_mass_matrix(u, x_aux[1])

def J_aux_callback(J_aux):
    Jd, Jr, relerrs, relerrs_r = J_aux

    s = '\n'
    s += 'Jd=' + "{:<10.2e}".format(Jd)
    s += ', Jr=' + "{:<10.2e}".format(Jr)
    s += '\nRelative errors forward:\n'
    for ii in range(len(relerrs)):
        s += "{:<10.2e}".format(relerrs[ii])
    s += '\nRelative errors reverse:\n'
    for ii in range(len(relerrs_r)):
        s += "{:<10.2e}".format(relerrs_r[ii])
    s += '\n'
    print(s)




