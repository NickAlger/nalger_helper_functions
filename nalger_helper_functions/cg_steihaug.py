import numpy as np
import typing as typ

import nalger_helper_functions.tree_linalg as tla

Vec    = typ.TypeVar('Vec')
Covec  = typ.TypeVar('Covec')
Scalar = typ.TypeVar('Scalar')


def cg_steihaug(
        hessian_matvec:         typ.Callable[[Vec], Covec], # u_vec -> H @ u_vec
        gradient:               Covec,
        trust_region_radius:    Scalar,
        rtol:                   Scalar,
        preconditioner_apply:   typ.Callable[[Vec],           Covec]  = None, # u    -> M @ u
        preconditioner_solve:   typ.Callable[[Covec],         Vec]    = None, # b    -> M^-1 @ b
        add_vectors:            typ.Callable[[Vec,      Vec], Vec]    = None, # u, v -> u + v
        add_covectors:          typ.Callable[[Covec,  Covec], Covec]  = None, # b, c -> b + c
        scale_vector:           typ.Callable[[Vec,   Scalar], Vec]    = None, # u, s -> s * u
        scale_covector:         typ.Callable[[Covec, Scalar], Covec]  = None, # b, s -> s * b
        dual_pairing:           typ.Callable[[Covec,    Vec], Scalar] = None, # b, u -> b(u)
        vector_is_bad:          typ.Callable[[Vec],           bool]   = None, # returns true if vector is bad (e.g., contains nans or infs)
        max_iter:   int  = 250,
        display:    bool = True,
        callback:   typ.Callable = None, #used as callback(xk), where xk is the current iterate
) -> typ.Tuple[
    typ.Any, # optimal search direction. same type as gradient
    typ.Tuple[
        int, # number of iterations
        str, # termination reason
    ]
]:
    '''Preconditioned variant of Algorithm 7.2 in Nocedal and Wright, page 171. See also: equation 4.5 on page 69.
    Approximately solves quadratic minimization problem:
        argmin_p f + g^T x + 0.5 * x^T H x

        See also:
            https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
    '''
    def _print(s: str):
        if display:
            print(s)

    callback                = (lambda z: None)  if callback             is None else callback
    add_vectors             = tla.add           if add_vectors          is None else add_vectors
    add_covectors           = tla.add           if add_covectors        is None else add_covectors
    scale_vector            = tla.scale         if scale_vector         is None else scale_vector
    scale_covector          = tla.scale         if scale_covector       is None else scale_covector
    preconditioner_apply    = (lambda u: u)     if preconditioner_apply is None else preconditioner_apply
    preconditioner_solve    = (lambda u: u)     if preconditioner_solve is None else preconditioner_solve
    dual_pairing            = tla.dot           if dual_pairing         is None else dual_pairing
    vector_is_bad           = tla.isbad         if vector_is_bad        is None else vector_is_bad

    g_covec = gradient

    r_covec = scale_covector(g_covec, -1.0)
    iM_r_vec = preconditioner_solve(r_covec)
    r0_iM_r0 = dual_pairing(r_covec, iM_r_vec)
    atol_squared = rtol**2 * r0_iM_r0

    x_vec = scale_vector(iM_r_vec, 0.0)

    p_vec = iM_r_vec

    r_iM_r = r0_iM_r0
    x_M_x = 0.0
    _print('initial ||r||_iM / ||r0||_iM=' + str(np.sqrt(r_iM_r / r0_iM_r0)))
    _print('initial ||x||_M=' + str(np.sqrt(x_M_x)))

    if r_iM_r < atol_squared:
        return p_vec, (0, 'tolerance_achieved')  # currently zero; take no steps

    for jj in range(1,max_iter+1):
        Hp_covec  = hessian_matvec(p_vec)
        pHp = dual_pairing(Hp_covec, p_vec)
        if pHp <= 0.0:
            if np.isinf(trust_region_radius):
                if jj == 0:
                    _print('Encountered negative curvature on first iteration. Probably due to numerical errors. dBd=' + str(pHp) + '. Infinite trust region radius: returning zero')
                    return x_vec, (jj + 1, 'gradient_points_downhill')
                else:
                    _print('Iterate ' + str(jj) + ' encountered negative curvature. dBd=' + str(pHp) + '. Infinite trust region radius: returning previous iterate.')
                    return x_vec, (jj + 1, 'encountered_negative_curvature')

            Mp_covec = preconditioner_apply(p_vec)
            p_M_p = dual_pairing(Mp_covec, p_vec)
            p_M_x = dual_pairing(Mp_covec, x_vec)
            tau1, tau2 = _interpolate_to_trust_region_boundary(x_M_x, p_M_p, p_M_x, trust_region_radius, display=True)

            x1_vec = add_vectors(x_vec, scale_vector(p_vec, tau1)) # x <- x + tau1*p
            x2_vec = add_vectors(x_vec, scale_vector(p_vec, tau2)) # x <- x + tau2*p
            m1 = dual_pairing(g_covec, x1_vec) + 0.5 * dual_pairing(x1_vec, hessian_matvec(x1_vec))
            m2 = dual_pairing(g_covec, x2_vec) + 0.5 * dual_pairing(x2_vec, hessian_matvec(x2_vec))
            if m1 <= m2:
                tau = tau1
                new_x_vec = x1_vec
                if vector_is_bad(new_x_vec):
                    print('INF or NAN detected. Terminating CG-Steihaug early')
                    return x_vec, (jj + 1, 'INF OR NAN')
                else:
                    x_vec = new_x_vec
            else:
                tau = tau2
                new_x_vec = x2_vec
                if vector_is_bad(new_x_vec):
                    print('INF or NAN detected. Terminating CG-Steihaug early')
                    return x_vec, (jj + 1, 'INF OR NAN')
                else:
                    x_vec = new_x_vec
            _print('m1=' + str(m1) + ', m2=' + str(m2))
            _print('Iterate ' + str(jj) + ' encountered negative curvature. tau=' + str(tau))
            return x_vec, (jj+1, 'encountered_negative_curvature')

        alpha = r_iM_r / pHp
        new_x_vec = add_vectors(x_vec, scale_vector(p_vec, alpha)) # x <- x + alpha*p
        if vector_is_bad(new_x_vec):
            print('INF or NAN detected. Terminating CG-Steihaug early')
            return x_vec, (jj + 1, 'INF OR NAN')
        else:
            x_vec = new_x_vec

        x_M_x = dual_pairing(preconditioner_apply(x_vec), x_vec)

        if x_M_x >= trust_region_radius ** 2:
            Mp_covec = preconditioner_apply(p_vec)
            p_M_p = dual_pairing(Mp_covec, p_vec)
            p_M_x = dual_pairing(Mp_covec, x_vec)
            tau, _ = _interpolate_to_trust_region_boundary(x_M_x, p_M_p, p_M_x, trust_region_radius, display=True)

            new_x_vec = add_vectors(x_vec, scale_vector(p_vec, tau)) # p = z + tau*d
            if vector_is_bad(new_x_vec):
                print('INF or NAN detected. Terminating early')
                return x_vec, (jj + 1, 'INF OR NAN')
            else:
                x_vec = new_x_vec

            _print(
                'Iterate ' + str(jj)
                + ' exited trust region. trust_region_radius=' + str(trust_region_radius)
                + ', tau=' + str(tau)
            )
            return x_vec, (jj+1, 'exited_trust_region')

        r_covec = add_covectors(r_covec, scale_covector(Hp_covec, -alpha)) # r <- r - alpha*Hp
        iM_r_vec = preconditioner_solve(r_covec)
        r_iM_r_previous = r_iM_r
        r_iM_r = dual_pairing(r_covec, iM_r_vec) # scalar, shape=()
        # print('ASDF r0_iM_r0=', r0_iM_r0)
        # print('ASDF r_iM_r=', r_iM_r)
        _print(
            'CG Iter:' + str(jj)
            + ', ||r||_iM / ||r0||_iM=' + str(np.sqrt(r_iM_r / r0_iM_r0))
            + ', ||x||_M=' + str(np.sqrt(x_M_x))
            + ', Delta=' + str(trust_region_radius)
        )
        if r_iM_r < atol_squared:
            _print('rtol=' + str(rtol) + ' achieved. ||r||/||g||=' + str(np.sqrt(r_iM_r / r0_iM_r0)))
            callback(x_vec)
            return x_vec, (jj+1, 'tolerance_achieved')

        beta = r_iM_r / r_iM_r_previous # scalar, shape=()
        p_vec = add_vectors(iM_r_vec, scale_vector(p_vec, beta))

        callback(x_vec)

    _print('Reached max_iter=' + str(max_iter) + ' without converging or exiting trust region.')

    return x_vec, (max_iter, 'performed_maximum_iterations')


def _interpolate_to_trust_region_boundary(
        x_M_x, # scalar
        p_M_p, # scalar
        p_M_x, # scalar
        trust_region_radius:    Scalar, # shape=()
        display:                bool,
) -> typ.Tuple[Scalar, Scalar]: # (tau1, tau2), scalars, elm_shape=()
    '''Find tau such that:
        Delta^2 = ||x + tau*p||^2
                = ||x||^2 + tau*2*(x, p) + tau^2*||p||^2
    which is:,
      tau^2 * ||p||^2 + tau * 2*(x, p) + ||x||^2-Delta^2 = 0
              |--a--|         |--b---|   |------c------|
    Note that c is negative since we were inside trust region last step
    '''
    a = p_M_p
    b = 2.0 * p_M_x
    c = x_M_x - trust_region_radius ** 2

    tau1 = (-b + np.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)  # quadratic formula, first root
    tau2 = (-b - np.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)  # quadratic formula, second root
    if display:
        print('a=', a, ', b=', b, ', c=', c, ', tau1=', tau1, ', tau2=', tau2)
    return tau1, tau2

