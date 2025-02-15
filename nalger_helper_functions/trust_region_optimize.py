import numpy as np
import jax.numpy as jnp
import typing as typ

from nalger_helper_functions.cg_steihaug import cg_steihaug, Vec, Covec, Scalar


Point   = typ.Any # type of point on manifold
Xaux    = typ.Any # type of auxiliary data computed for a base point x
Jaux    = typ.Any # type of auxiliary data returned when computing objective function
Gaux    = typ.Any # type of auxiliary data returned when computing gradient

def trust_region_optimize(
        objective:      typ.Callable[[Point, Xaux],                  typ.Tuple[Scalar, Jaux]], # x, x_aux              -> J(x), J_aux
        gradient:       typ.Callable[[Point, Xaux, Jaux],            typ.Tuple[Covec, Gaux]],  # x, x_aux, J_aux       -> g(x), g_aux
        hessian_matvec: typ.Callable[[Vec, Point, Xaux, Jaux, Gaux], Covec],                   # u, x, x, J_aux, g_aux -> H(x) @ u
        x0:             Point, # initial guess
        #
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
        #
        compute_x_aux:        typ.Callable[[Point],                            Xaux]   = None, # x -> x_aux
        preconditioner_apply: typ.Callable[[Vec,   Point,  Xaux,  Jaux, Gaux], Covec]  = None, # u, x, x_aux, J_aux, G_aux -> M(x) @ u
        preconditioner_solve: typ.Callable[[Covec, Point,  Xaux,  Jaux, Gaux], Vec]    = None, # b, x, x_aux, J_aux, G_aux -> M(x)^-1 @ b
        add_vectors:          typ.Callable[[Vec,   Vec,    Point, Xaux],       Vec]    = None, # u, v, x, x_aux -> u + v
        add_covectors:        typ.Callable[[Covec, Covec,  Point, Xaux],       Covec]  = None, # b, c, x, x_aux -> b + c
        scale_vector:         typ.Callable[[Vec,   Scalar, Point, Xaux],       Vec]    = None, # u, s, x, x_aux -> s * u
        scale_covector:       typ.Callable[[Covec, Scalar, Point, Xaux],       Covec]  = None, # b, s, x, x_aux -> s * b
        dual_pairing:         typ.Callable[[Covec, Vec,    Point, Xaux],       Scalar] = None, # b, u, x, x_aux -> b(u)
        retract:              typ.Callable[[Point, Vec,    Xaux],              Point]  = None, # x, u, x_aux -> x (+) u_vec: retract vector from tangent plane to manifold
        #
        newton_callback: typ.Callable[[Point], typ.Any] = None, # used as newton_callback(x), where x is the current newton iterate
        cg_callback:     typ.Callable[[Vec],   typ.Any] = None, # used as cg_callback(p), where p is the current cg iterate
        x_aux_callback:  typ.Callable[[Xaux],  typ.Any] = None, # used as x_aux_callback(x_aux), where x_aux = compute_x_aux(x)
        J_aux_callback:  typ.Callable[[Jaux],  typ.Any] = None, # used as J_aux_callback(J_aux), where J, J_aux = objective(x)
        g_aux_callback:  typ.Callable[[Gaux],  typ.Any] = None, # used as g_aux_callback(g_aux), where g, g_aux = objective(x, J_aux)
):
    def _print(*args, **kwargs):
        if newton_display:
            print(*args, **kwargs)

    newton_max_steps = newton_max_iter if newton_max_steps is None else newton_max_steps

    preconditioner_apply    = (lambda u, x, x_aux, J_aux, g_aux: u)           if preconditioner_apply is None else preconditioner_apply
    preconditioner_solve    = (lambda w, x, x_aux, J_aux, g_aux: w)           if preconditioner_solve is None else preconditioner_solve
    dual_pairing            = (lambda w, u, x,     x_aux:        np.sum(w*u)) if dual_pairing         is None else dual_pairing
    add_vectors             = (lambda u, v, x,     x_aux:        u + v)       if add_vectors          is None else add_vectors
    add_covectors           = (lambda u, v, x,     x_aux:        u + v)       if add_covectors        is None else add_covectors
    scale_vector            = (lambda u, c, x,     x_aux:        c*u)         if scale_vector         is None else scale_vector
    scale_covector          = (lambda u, c, x,     x_aux:        c*u)         if scale_covector       is None else scale_covector
    retract                 = (lambda x, u, x_aux:               x + u)       if retract              is None else retract

    null_func = lambda x: None
    null_func_if_none = lambda func: null_func if func is None else func

    newton_callback     = null_func_if_none(newton_callback)
    cg_callback         = null_func_if_none(cg_callback)
    compute_x_aux       = null_func_if_none(compute_x_aux)
    x_aux_callback      = null_func_if_none(x_aux_callback)
    J_aux_callback      = null_func_if_none(J_aux_callback)
    g_aux_callback      = null_func_if_none(g_aux_callback)

    x = x0
    x_aux = compute_x_aux(x)
    J, J_aux = objective(x, x_aux)
    g, g_aux = gradient(x, x_aux, J_aux)
    iM_g = preconditioner_solve(g, x, x_aux, J_aux, g_aux)
    g_iM_g = dual_pairing(g, iM_g, x, x_aux)

    g0_iM_g0 = g_iM_g

    _print('J0=' + str(J))
    _print('||g0||_iM=' + str(np.sqrt(g_iM_g)))
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

        hvp_func            = lambda p:     hessian_matvec(p, x, x_aux, J_aux, g_aux)
        cg_dual_pairing     = lambda w, u:  dual_pairing(w, u, x, x_aux)
        cg_add_vectors      = lambda u, v:  add_vectors(u, v, x, x_aux)
        cg_add_covectors    = lambda u, v:  add_covectors(u, v, x, x_aux)
        cg_scale_vector     = lambda u, c:  scale_vector(u, c, x, x_aux)
        cg_scale_covector   = lambda u, c:  scale_covector(u, c, x, x_aux)
        cg_preconditioner_apply = lambda u: preconditioner_apply(u, x, x_aux, J_aux, g_aux)
        cg_preconditioner_solve = lambda w: preconditioner_solve(w, x, x_aux, J_aux, g_aux)

        cg_max_iter2 = cg_max_iter
        if newton_iter == 1:
            cg_max_iter2 = 1
            trust_region_radius = np.inf

        cg_rtol = np.maximum(np.minimum(0.5, np.power(np.sqrt(g_iM_g / g0_iM_g0), cg_rtol_power)), newton_rtol / 3.0)
        p, (num_cg_iter, termination_reason) = cg_steihaug(
            hvp_func, g, trust_region_radius, cg_rtol,
            preconditioner_apply=cg_preconditioner_apply,
            preconditioner_solve=cg_preconditioner_solve,
            add_vectors=cg_add_vectors,
            add_covectors=cg_add_covectors,
            scale_vector=cg_scale_vector,
            scale_covector=cg_scale_covector,
            dual_pairing=cg_dual_pairing,
            max_iter=cg_max_iter2, display=cg_display, callback=cg_callback,
        )

        if newton_iter == 1:
            termination_reason = 'first_iteration'
            num_cg_iter = 1
            cg_rtol = jnp.array(1.0)
            M_p = cg_preconditioner_apply(p)
            trust_region_radius = cg_dual_pairing(M_p, p)
            min_trust_region_radius = trust_region_min_radius_factor * trust_region_radius

        s = '{:<12s}'.format('trust size') + '{:<10s}'.format('CG rtol') + '{:<10s}'.format('#CG') + 'CG termination reason' + '\n'
        s += '{:<12.1E}'.format(trust_region_radius) + '{:<10.1E}'.format(cg_rtol) + '{:<10d}'.format(num_cg_iter) + termination_reason + '\n'
        _print(s)

        predicted_J = J + cg_dual_pairing(g, p) + 0.5 * cg_dual_pairing(hvp_func(p), p)

        previous_step = (x, p, x_aux)
        new_x = retract(x, p, x_aux)

        new_x_aux = compute_x_aux(new_x)
        new_J, new_J_aux = objective(new_x, new_x_aux)

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
            g, g_aux = gradient(x, x_aux, J_aux)
            iM_g = preconditioner_solve(g, x, x_aux, J_aux, g_aux)
            g_iM_g = dual_pairing(g, iM_g, x, x_aux)
            newton_steps_taken += 1
        else:
            _print('Keeping x the same.\n')

        _print('||g||/||g0||=' + str(np.sqrt(g_iM_g / g0_iM_g0)))
        newton_callback(x_aux)
        x_aux_callback(x_aux)
        J_aux_callback(J_aux)
        g_aux_callback(g_aux)

        end_newton_iter = False
        if trust_region_radius <= min_trust_region_radius or np.isnan(trust_region_radius):
            print('Trust region too small: ending Newton iteration.')
            end_newton_iter = True

        if newton_iter + 1 >= newton_min_iter:
            if g_iM_g <= g0_iM_g0 * newton_rtol**2:
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

