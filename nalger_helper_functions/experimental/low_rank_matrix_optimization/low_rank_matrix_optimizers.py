import numpy as np
import jax.numpy as jnp
import jax
import typing as typ
import functools as ft

import nalger_helper_functions.experimental.low_rank_matrix_optimization.low_rank_matrix_optimization_problem as lrmop
import nalger_helper_functions.experimental.low_rank_matrix_optimization.low_rank_matrix_manifold as lrmm
from nalger_helper_functions import cg_steihaug, trust_region_optimize
import nalger_helper_functions.tree_linalg as tla
from nalger_helper_functions.rsvd import rsvd_double_pass

# jax.config.update("jax_enable_x64", True) # enable double precision


__all__ = [
    'low_rank_manifold_trust_region_optimize_fixed_rank',
    'svd_initial_guess',
    'low_rank_manifold_trust_region_optimize_fixed_rank_nonlinear',
]


Inputs  = typ.TypeVar('Inputs')
Outputs = typ.TypeVar('Outputs')
Param   = typ.TypeVar('Param')
Scalar  = typ.TypeVar('Scalar')

def low_rank_manifold_trust_region_optimize_fixed_rank_nonlinear(
        inputs:         Inputs,
        true_outputs:   Outputs,
        m0:             Param,
        forward_map: typ.Callable[[Param, Inputs], Outputs],
        loss_func: typ.Callable[
            [
                Outputs, # y
                Outputs, # y_true
            ],
            typ.Tuple[
                Scalar, # Jd
                typ.Tuple[
                    jnp.ndarray, # relerrs
                    jnp.ndarray, # relerrs_r
                ],
            ]
        ],
        regularization_func: typ.Callable[
            [Param], # m
            Scalar,  # Jr
        ],
        **kwargs,
):
    @jax.jit
    def _J_func(
            m: Param,
            m_aux, # not used
    ):
        y: Outputs = forward_map(m, inputs)
        Jd, (relerrs, relerrs_r) = loss_func(y, true_outputs)
        Jr = regularization_func(m)
        J = Jd + Jr
        return J, (Jd, Jr, y, (relerrs, relerrs_r))

    _g0d_func = jax.grad(_J_func, argnums=0, has_aux=True)
    _g0r_func = jax.grad(regularization_func)

    @jax.jit
    def _g_func(
            m: Param,
            m_aux,
            J_aux,
    ):
        g0d, g_aux = _g0d_func(m, m_aux)
        g0r = _g0r_func(m)
        g0 = tla.add(g0d, g0r)
        g1 = lrmm.tangent_orthogonal_projection(m, g0)
        return g1, (g0d, g0r, g_aux)

    _forward_map2 = lambda q: forward_map(q, inputs)

    @jax.jit
    def _forward_map_jvp(
            p: Param,
            m: Param,
    ):
        return jax.jvp(_forward_map2, (m,), (p,))[1]

    _forward_map_vjp = jax.jit(lambda dy: jax.vjp(_forward_map2, m0)[1](dy)[0])

    _loss2 = lambda y: loss_func(y, true_outputs)[0]

    _loss2_grad = jax.grad(_loss2, argnums=0, has_aux=False)

    @jax.jit
    def _loss_hvp(
            dy:     Outputs,
            y:      Outputs,
    ):
        return jax.jvp(_loss2_grad, (y,), (dy,))[1]

    _reg_grad = jax.grad(regularization_func, argnums=0, has_aux=False)

    @jax.jit
    def _reg_hvp(
            p: Param,
            m: Param,
    ):
        return jax.jvp(_reg_grad, (m,), (p,))[1]

    @jax.jit
    def _H_matvec_func(
            p: Param,
            m: Param,
            x_aux,
            J_aux1,
            g_aux,
    ):
        _, _, y, _ = J_aux1
        p2 = lrmm.tangent_orthogonal_projection(m, p)

        dy = _forward_map_jvp(p, m)
        dy2 = _loss_hvp(dy, y)
        Hd_p = _forward_map_vjp(dy2)

        Hr_p = _reg_hvp(p2, m)

        H_p0 = tla.add(Hd_p, Hr_p)

        H_p1 = lrmm.tangent_orthogonal_projection(m, H_p0)
        return H_p1

    _apply_M_func   = lambda p, m,      m_aux, J_aux, g_aux:    preconditioner_apply(p, m, m_aux)
    _solve_M_func   = lambda p, m,      m_aux, J_aux, g_aux:    preconditioner_solve(p, m, m_aux)
    _retract_func   = lambda m, p,      m_aux:                  projected_retract(m, p)

    def _J_aux_callback(J_aux):
        Jd, Jr, y, (relerrs, relerrs_r) = J_aux
        print_relerrs(Jd, Jr, relerrs, relerrs_r)

    return trust_region_optimize(
        _J_func,
        _g_func,
        _H_matvec_func,
        m0,
        retract=_retract_func,
        preconditioner_apply=_apply_M_func,
        preconditioner_solve=_solve_M_func,
        compute_x_aux=compute_x_aux,
        J_aux_callback=_J_aux_callback,
        **kwargs,
    )


def low_rank_manifold_trust_region_optimize_fixed_rank(
        inputs,
        true_outputs,
        x0,
        apply_P: typ.Callable[[typ.Tuple[jnp.ndarray, jnp.ndarray]], typ.Tuple[jnp.ndarray, jnp.ndarray]] = lambda u: u,
        apply_R: typ.Callable[[typ.Tuple[jnp.ndarray, jnp.ndarray]], typ.Tuple[jnp.ndarray, jnp.ndarray]] = lambda u: tla.scale(u, 0.0),
        **kwargs,
):
    _J_func         = lambda x, x_aux:                          lrmop.objective(x, inputs, true_outputs, apply_P, apply_R)
    _g_func         = lambda x, x_aux,  J_aux:                  projected_gradient(x, J_aux, inputs, true_outputs, apply_P, apply_R)
    _H_matvec_func  = lambda p, x,      x_aux, J_aux, g_aux:    projected_hessian_matvec(x, p, J_aux, inputs, true_outputs, apply_P, apply_R)
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
    return lrmm.left_orthogonalize_low_rank(x0)


@jax.jit
def compute_x_aux(x):
    M_helper = lrmm.make_tangent_mass_matrix_helper(x)
    iM_helper = jnp.linalg.inv(M_helper)
    return M_helper, iM_helper


def projected_gradient(
        # arguments used by optimizer:
        x, J_aux,
        # arguments removed by partial application:
        inputs, true_outputs, apply_P, apply_R,
):
    _, _, y, _ = J_aux
    g0, g_aux = lrmop.gradient(x, inputs, y, true_outputs, apply_P, apply_R)
    g1 = lrmm.tangent_orthogonal_projection(x, g0)
    return g1, g_aux


def projected_hessian_matvec(
        # arguments used by optimizer
        x, p, J_aux,
        # arguments removed by partial application
        inputs, true_outputs, apply_P, apply_R,
):
    Jd, Jr, outputs, Jd_aux = J_aux
    p2 = lrmm.tangent_orthogonal_projection(x, p)
    Hp0, _ = lrmop.gauss_newton_hessian_matvec(p2, x, inputs, outputs, true_outputs, apply_P, apply_R)
    Hp1 = lrmm.tangent_orthogonal_projection(x, Hp0)
    return Hp1


def projected_retract_arbitrary_rank(
        x, p, rank,
):
    p2 = lrmm.tangent_orthogonal_projection(x, p)
    x_plus_p = lrmm.retract_tangent_vector(x, p2, rank)
    return x_plus_p

projected_retract = lambda x, p: projected_retract_arbitrary_rank(x, p, None)

preconditioner_apply = lambda u, x, x_aux: lrmm.apply_tangent_mass_matrix(u, x_aux[0])
preconditioner_solve = lambda u, x, x_aux: lrmm.apply_tangent_mass_matrix(u, x_aux[1])

def print_relerrs(Jd, Jr, relerrs, relerrs_r):
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

def J_aux_callback(J_aux):
    Jd, Jr, outputs, (relerrs, relerrs_r) = J_aux
    print_relerrs(Jd, Jr, relerrs, relerrs_r)




