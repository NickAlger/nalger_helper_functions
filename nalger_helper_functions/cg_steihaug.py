import jax.numpy as jnp
import typing as typ


def cg_steihaug(
        hessian_matvec:         typ.Callable[[typ.Any], typ.Any], # u -> H @ u
        gradient:               typ.Any,
        add:            typ.Callable[[typ.Any, typ.Any], typ.Any],    # (u_vector, v_vector) -> u_vector + v_vector
        scale:          typ.Callable[[typ.Any, typ.Any], typ.Any],    # (u_vector, c_scalar) -> c_scalar * u_vector
        inner_product:  typ.Callable[[typ.Any, typ.Any], typ.Any],    # (u_vector, v_vector) -> <u_vector, v_vector>
        trust_region_radius,    # scalar
        rtol,                   # scalar
        max_iter:   int  = 250,
        display:    bool = True,
        callback:   typ.Callable[[typ.Any], typ.Any] = None, #used as callback(zk), where zk is the current iterate
) -> typ.Tuple[
    typ.Any, # optimal search direction. same type as gradient
    typ.Tuple[
        int, # number of iterations
        str, # termination reason
    ]
]:
    '''Algorithm 7.2 in Nocedal and Wright, page 171. See also: equation 4.5 on page 69.
    Approximately solves quadratic minimization problem:
        argmin_p f + g^T p + 0.5 * p^T H P
    '''
    def _print(s: str):
        if display:
            print(s)

    if callback is None:
        callback = lambda z: None

    gradnorm_squared = inner_product(gradient, gradient)
    atol_squared = rtol**2 * gradnorm_squared

    z = scale(gradient, 0.0)
    r = gradient
    d = scale(gradient, -1.0)

    z_normsquared = 0.0
    r_normsquared = inner_product(r, r)
    _print('initial ||r||/||g||=' + str(jnp.sqrt(r_normsquared / gradnorm_squared)))

    if r_normsquared < atol_squared:
        return z, (0, 'tolerance_achieved')  # currently zero; take no steps

    for jj in range(max_iter):
        Bd  = hessian_matvec(d)
        dBd = inner_product(d, Bd)
        if dBd <= 0.0:
            if np.isinf(trust_region_radius):
                if jj == 0:
                    p = gradient
                    _print('gradient points downhill. Probably due to numerical errors. dBd=' + str(dBd))
                    return p, (jj + 1, 'gradient_points_downhill')
                else:
                    p = z
                    _print('Iterate ' + str(jj) + ' encountered negative curvature. dBd=' + str(dBd))
                    return p, (jj + 1, 'encountered_negative_curvature')

            tau1, tau2 = _interpolate_to_trust_region_boundary(z, d, z_normsquared, trust_region_radius, inner_product, display=True)
            p1 = add(z, scale(d, tau1)) # p = z + tau1*d
            p2 = add(z, scale(d, tau2)) # p = z + tau2*d
            m1 = inner_product(gradient, p1) + 0.5 * inner_product(p1, hessian_matvec(p1))
            m2 = inner_product(gradient, p2) + 0.5 * inner_product(p2, hessian_matvec(p2))
            if m1 <= m2:
                tau = tau1
                p = p1
            else:
                tau = tau2
                p = p2
            _print('m1=' + str(m1) + ', m2=' + str(m2))
            _print('Iterate ' + str(jj) + ' encountered negative curvature. tau=' + str(tau))
            return p, (jj+1, 'encountered_negative_curvature')

        alpha = r_normsquared / dBd
        z_next = add(z, scale(d, alpha)) # z_next = z + alpha*d
        z_next_normsquared = inner_product(z_next, z_next)

        if z_next_normsquared >= trust_region_radius ** 2:
            tau, _ = _interpolate_to_trust_region_boundary(z, d, z_normsquared, trust_region_radius, inner_product, display)
            p = add(z, scale(d, tau)) # p = z + tau*d
            _print(
                'Iterate ' + str(jj)
                + ' exited trust region. trust_region_radius=' + str(trust_region_radius)
                + ', tau=' + str(tau)
            )
            return p, (jj+1, 'exited_trust_region')

        r_next = add(r, scale(Bd, alpha)) # r_next = r + alpha*Bd
        r_next_normsquared: jnp.ndarray = inner_product(r_next, r_next) # scalar, shape=()
        _print(
            'CG Iter:' + str(jj)
            + ', ||r||/||g||=' + str(jnp.sqrt(jnp.sqrt(r_next_normsquared / gradnorm_squared)))
            + ', ||z||=' + str(jnp.sqrt(z_next_normsquared))
            + ', Delta=' + str(trust_region_radius)
        )
        if r_next_normsquared < atol_squared:
            _print('rtol=' + str(rtol) + ' achieved. ||r||/||g||=' + str(jnp.sqrt(r_next_normsquared / gradnorm_squared)))
            callback(z_next)
            return z_next, (jj+1, 'tolerance_achieved')

        beta: jnp.ndarray = r_next_normsquared / r_normsquared # scalar, shape=()
        d = add(scale(r_next, -1.0), scale(d, beta))

        r = r_next
        z = z_next
        callback(z)
        z_normsquared = z_next_normsquared
        r_normsquared = r_next_normsquared

    _print('Reached max_iter=' + str(max_iter) + ' without converging or exiting trust region.')

    return z, (max_iter, 'performed_maximum_iterations')


def _interpolate_to_trust_region_boundary(
        z:                      typ.Any,
        d:                      typ.Any,
        z_normsquared:          jnp.ndarray, # scalar, shape=()
        trust_region_radius:    jnp.ndarray, # shape=()
        inner_product:          typ.Callable[[typ.Any, typ.Any], typ.Union[float, jnp.ndarray]],  # (u, v) -> <u,v>
        display:                bool,
) -> typ.Tuple[jnp.ndarray, jnp.ndarray]: # (tau1, tau2), scalars, elm_shape=()
    '''Find tau such that:
        Delta^2 = ||z + tau*d||^2
                = ||z||^2 + tau*2*(z, d) + tau^2*||d||^2
    which is:,
      tau^2 * ||d||^2 + tau * 2*(z, d) + ||z||^2-Delta^2 = 0
              |--a--|         |--b---|   |------c------|
    Note that c is negative since we were inside trust region last step
    '''
    d_normsquared = inner_product(d, d)
    z_dot_d = inner_product(z, d)

    a = d_normsquared
    b = 2.0 * z_dot_d
    c = z_normsquared - trust_region_radius ** 2

    tau1 = (-b + jnp.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)  # quadratic formula, first root
    tau2 = (-b - jnp.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)  # quadratic formula, second root
    if display:
        print('a=', a, ', b=', b, ', c=', c, ', tau1=', tau1, ', tau2=', tau2)
    return tau1, tau2

