import numpy as np
import jax.numpy as jnp
import typing as typ

from nalger_helper_functions.cg_steihaug import cg_steihaug


def trust_region_optimize(
        objective_func:         typ.Callable, # (x, x_aux)                  -> (J(x), J_aux)
        gradient_func:          typ.Callable, # (x, x_aux, J_aux)           -> (g(x), g_aux)
        hessian_matvec_func:    typ.Callable, # (x, p, x_aux, J_aux, g_aux) -> H(m) @ p
        x0:             typ.Any, # initial guess
        add:            typ.Callable,  # (u_vector, v_vector,  x, x_aux) -> u_vector + v_vector
        retract:        typ.Callable,  # (x_primal, u_tangent, x_aux)    -> x_primal (+) u_tangent: retract vector from tangent plane to manifold
        scale:          typ.Callable,  # (u_vector, c_scalar,  x, x_aux) -> c_scalar * u_vector
        inner_product:  typ.Callable,  # (u_vector, v_vector,  x, x_aux) -> <u_vector, v_vector>
        newton_rtol:        float = 1e-5,
        cg_rtol_power:      float = 0.5, # between 0 and 1
        cg_max_iter:        int  = 250,
        newton_min_iter:    int = 1,
        newton_max_iter:    int = 50, # counts all iterations, whether or not we take a step
        newton_max_steps:   int = None, # counts only iterations where a step is taken
        trust_region_min_radius_factor:         float = 1e-8,
        trust_region_max_radius:                float = np.inf,
        trust_region_minimum_reduction_ratio:   float = 0.1,  # trust region parameter
        newton_display: bool = True,
        cg_display:     bool = True,
        newton_callback: typ.Callable = None, # used as newton_callback(x), where x is the current newton iterate
        cg_callback:     typ.Callable = None,  # used as cg_callback(z), where z is the current cg iterate
        compute_x_aux:   typ.Callable = None,  # x -> x_aux
        x_aux_callback:  typ.Callable = None, # used as x_aux_callback(x_aux), where x_aux = compute_x_aux(x)
        J_aux_callback:  typ.Callable = None, # used as J_aux_callback(J_aux), where J, J_aux = objective(x)
        g_aux_callback:  typ.Callable = None, # used as g_aux_callback(g_aux), where g, g_aux = objective(x, J_aux)
):
    def _print(*args, **kwargs):
        if newton_display:
            print(*args, **kwargs)

    newton_max_steps = newton_max_iter if newton_max_steps is None else newton_max_steps

    null_func = lambda x: None
    null_func_if_none = lambda func: null_func if func is None else func

    newton_callback     = null_func_if_none(newton_callback)
    cg_callback         = null_func_if_none(cg_callback)
    compute_x_aux       = null_func_if_none(compute_x_aux)
    x_aux_callback      = null_func_if_none(x_aux_callback)
    J_aux_callback      = null_func_if_none(J_aux_callback)
    g_aux_callback      = null_func_if_none(g_aux_callback)

    def _norm(u, x, x_aux):
        return np.sqrt(inner_product(u, u, x, x_aux))

    x0_aux = compute_x_aux(x0)
    J0, J0_aux = objective_func(x0, x0_aux)
    g0, g0_aux = gradient_func(x0, x0_aux, J0_aux)
    norm_g0 = _norm(g0, x0, x0_aux)

    x = x0
    x_aux = x0_aux
    J = J0
    J_aux = J0_aux
    g = g0
    g_aux = g0_aux
    norm_g = norm_g0

    _print('J0=' + str(J))
    _print('||g0||=' + str(norm_g))
    newton_callback(x_aux)
    x_aux_callback(x_aux)
    J_aux_callback(J_aux)
    g_aux_callback(g_aux)

    trust_region_radius = None
    min_trust_region_radius = None  # min_trust_region_radius_factor * trust_region_radius

    newton_steps_taken: int = 0
    g_reduction_achieved = False
    for newton_iter in range(1, newton_max_iter + 1):
        _print('{:<12s}'.format('Newton Iter: ') + str(newton_iter) + '\n')

        hvp_func = lambda p: hessian_matvec_func(x, p, x_aux, J_aux, g_aux)
        cg_add = lambda u, v: add(u, v, x, x_aux)
        cg_scale = lambda u, c: scale(u, c, x, x_aux)
        cg_inner_product = lambda u, v: inner_product(u, v, x, x_aux)

        if newton_iter == 1:
            Hg = hvp_func(g)
            a = inner_product(g, g, x, x_aux) / inner_product(g, Hg, x, x_aux)
            p = scale(g, -a, x, x_aux)
            termination_reason = 'first_iteration'
            num_cg_iter = 1
            cg_rtol = jnp.array(1.0)
            trust_region_radius = _norm(p, x, x_aux)
            min_trust_region_radius = trust_region_min_radius_factor * trust_region_radius
        else:
            cg_rtol = np.maximum(np.minimum(0.5, np.power(norm_g / norm_g0, cg_rtol_power)), newton_rtol / 3.0)
            # cg_rtol = 1e-1
            p, (num_cg_iter, termination_reason) = cg_steihaug(
                hvp_func, g, cg_add, cg_scale, cg_inner_product, trust_region_radius,
                cg_rtol, cg_max_iter, cg_display, cg_callback,
            )

        s = '{:<12s}'.format('trust size') + '{:<10s}'.format('CG rtol') + '{:<10s}'.format('#CG') + 'CG termination reason' + '\n'
        s += '{:<12.1E}'.format(trust_region_radius) + '{:<10.1E}'.format(cg_rtol) + '{:<10d}'.format(num_cg_iter) + termination_reason + '\n'
        _print(s)

        predicted_J = J + inner_product(g, p, x, x_aux) + 0.5 * inner_product(p, hvp_func(p), x, x_aux)

        previous_step = (x, p, x_aux)
        new_x = retract(x, p, x_aux)

        new_x_aux = compute_x_aux(new_x)
        new_J, new_J_aux = objective_func(new_x, new_x_aux)

        new_trust_region_radius, reduction_ratio = _update_trust_region_size(
            J, new_J, predicted_J, trust_region_radius,
            min_trust_region_radius, trust_region_max_radius, termination_reason, newton_display,
        )

        trust_region_radius = new_trust_region_radius
        if reduction_ratio > trust_region_minimum_reduction_ratio:
            _print('Updating x.\n')
            x = new_x
            x_aux = new_x_aux
            J = new_J
            J_aux = new_J_aux
            g, g_aux = gradient_func(x, x_aux, J_aux)
            norm_g = _norm(g, x, x_aux)
            newton_steps_taken += 1
        else:
            _print('Keeping x the same.\n')

        _print('J0=' + str(J))
        _print('||g||/||g0||=' + str(norm_g / norm_g0))
        newton_callback(x_aux)
        x_aux_callback(x_aux)
        J_aux_callback(J_aux)
        g_aux_callback(g_aux)

        end_newton_iter = False
        if trust_region_radius <= min_trust_region_radius or np.isnan(trust_region_radius):
            print('Trust region too small: ending Newton iteration.')
            end_newton_iter = True

        if newton_iter + 1 >= newton_min_iter:
            if norm_g <= norm_g0 * newton_rtol:
                g_reduction_achieved = True
                end_newton_iter = True

        if newton_steps_taken >= newton_max_steps:
            print('newton_max_steps:',newton_max_steps, ', newton_steps_taken:', newton_steps_taken, ', ending Newton iteration.')
            end_newton_iter = True

        if end_newton_iter:
            break

    if g_reduction_achieved:
        _print('Achieved |g|/|g0| <= ' + str(newton_rtol) + '\n')

    return x, previous_step


def _update_trust_region_size(
        J_old,          # scalar. E.g., float or array with shape=()
        J_true,         # scalar
        J_predicted,    # scalar
        old_trust_region_radius, # scalar
        min_trust_region_radius, # scalar
        max_trust_region_radius, # scalar
        termination_reason: str,
        display:            bool = True,
) -> typ.Tuple[
    typ.Any, # scalar. new_trust_region_radius
    typ.Any, # scalar. reduction_ratio
]:
    def _printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    J_old = np.array(J_old) # increase precision; (x-y)/(x-z) has big rounding errors when x, y, z are all close
    J_true = np.array(J_true)
    J_predicted = np.array(J_predicted)
    actual_drop = J_old - J_true
    predicted_drop = J_old - J_predicted
    reduction_ratio = actual_drop / predicted_drop
    relative_reduction = actual_drop / J_true
    s = '{:<15s}'.format('Jd_old') + '{:<15s}'.format('Jd_true') + '{:<15s}'.format('Jd_predicted') + '\n'
    s += '{:<15.5E}'.format(J_old) + '{:<15.5E}'.format(J_true) + '{:<15.5E}'.format(J_predicted) + '\n'
    _printmaybe(s)
    _printmaybe('Drop in Jd from previous Newton iteration:')
    s = '{:<12s}'.format('relative') + '{:<12s}'.format('predicted') + '{:<12s}'.format(
        'actual') + '{:<18s}'.format('predicted/actual') + '\n'
    s += '{:<12.1E}'.format(relative_reduction) + '{:<12.1E}'.format(predicted_drop) + '{:<12.1E}'.format(
        actual_drop) + '{:<18.1E}'.format(reduction_ratio) + '\n'
    _printmaybe(s)

    # Modification of Algorithm 4.1 in Nocedal and Wright, page 69.
    # if reduction_ratio < 0.25:
    if (0.75 * J_old < J_true - 0.25 * J_predicted) or (J_true < 0.0) or (J_predicted < 0.0): # Mathematically equivalent but more numerically stable
        new_trust_region_radius = jnp.maximum(0.25 * old_trust_region_radius, min_trust_region_radius)
        s = ('reducing trust region size: '
             + '{:>8.1E}'.format(old_trust_region_radius)
             + ' -> '
             + '{:<8.1E}'.format(new_trust_region_radius)
             + '\n')
        _printmaybe(s)
    else:
        # if reduction_ratio > 0.75 and termination_reason.lower() in {
        #     'exited_trust_region', 'encountered_negative_curvature'
        # }:
        if 0.25 * J_old > J_true - 0.75 * J_predicted and termination_reason.lower() in {
            'exited_trust_region', 'encountered_negative_curvature'
        }:
            new_trust_region_radius = jnp.minimum(2.0 * old_trust_region_radius, max_trust_region_radius)
            s = ('increasing trust region size: ' +
                 '{:>8.1E}'.format(old_trust_region_radius) +
                 ' -> ' +
                 '{:<8.1E}'.format(new_trust_region_radius) +
                 '\n')
            _printmaybe(s)
        else:
            new_trust_region_radius = old_trust_region_radius
            s = ('keeping same trust region size: '
                 + '{:>8.1E}'.format(new_trust_region_radius)
                 )
            _printmaybe(s)

    return new_trust_region_radius, reduction_ratio

