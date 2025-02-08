import numpy as np
import jax.numpy as jnp
import jax
import typing as typ

import matplotlib.pyplot as plt


@jax.jit
def base_to_full(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ]
) -> jnp.ndarray: # XY, shape=(N,m)
    X, Y = base
    return X @ Y


@jax.jit
def left_orthogonalize_base(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray, # Q, shape=(N,r), Q^T Q = I
    jnp.ndarray, # Y2
]:
    '''QZ = XY. columns of Q are orthonormal'''
    X, Y = base
    Q, R = jnp.linalg.qr(X, mode='reduced')
    Y2 = R @ Y
    left_orthogonal_base = (Q, Y2)
    return left_orthogonal_base


def tangent_vector_to_full(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
) -> jnp.ndarray:
    X, Y = base
    dX, dY = perturbation
    return base_to_full((dX, Y)) + base_to_full((X, dY))


@jax.jit
def add_tangent_vectors(
        perturbation1: typ.Tuple[
            jnp.ndarray, # dX1, shape=(N,r)
            jnp.ndarray, # dY1, shape=(r,M)
        ],
        perturbation2: typ.Tuple[
            jnp.ndarray,  # dX2, shape=(N,r)
            jnp.ndarray,  # dY2, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # dX1 + dX2, shape=(N,r)
    jnp.ndarray,  # dY1 + dY2, shape=(r,M)
]:
    return perturbation1[0] + perturbation2[0], perturbation1[1] + perturbation2[1]


@jax.jit
def scale_tangent_vector(
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
        c: jnp.ndarray, # scalar, shape=()
) -> typ.Tuple[
    jnp.ndarray,  # dX1, shape=(N,r)
    jnp.ndarray,  # c*dY1, shape=(r,M)
]: # scaled_perturbation
    dX, dY = perturbation
    scaled_perturbation = (c*dX, c*dY)
    return scaled_perturbation


@jax.jit
def standardize_perturbation(
        left_orthogonal_base: typ.Tuple[
            jnp.ndarray,  # Q, shape=(N,r), Q^T Q = I
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX_perp, shape=(N,r)
            jnp.ndarray,  # dY2, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # dX2, shape=(N,r), Q^T dX2 = 0
    jnp.ndarray,  # dY2, shape=(r,M)
]: # standard_perturbation
    Q, Y = left_orthogonal_base
    dX, dY = perturbation
    C = Q.T @ dX
    dX_parallel = Q @ C
    dX_perp = dX - dX_parallel
    dY2 = dY + C @ Y
    standard_perturbation = (dX_perp, dY2)
    return standard_perturbation


@jax.jit
def make_inner_product_helper_matrix(
        left_orthogonal_base: typ.Tuple[
            jnp.ndarray,  # Q, shape=(N,r), Q^T Q = I
            jnp.ndarray,  # Y, shape=(r,M)
        ],
) -> jnp.ndarray: # inner_product_helper_matrix=YY^T, shape=(r,r)
    Q, Y = left_orthogonal_base
    return Y @ Y.T


@jax.jit
def inner_product_of_tangent_vectors(
        standard_perturbation1: typ.Tuple[
            jnp.ndarray,  # dX1_perp, shape=(N,r)
            jnp.ndarray,  # dY1, shape=(r,M)
        ],
        standard_perturbation2: typ.Tuple[
            jnp.ndarray,  # dX2_perp, shape=(N,r)
            jnp.ndarray,  # dY2, shape=(r,M)
        ],
        inner_product_helper_matrix: jnp.ndarray, # shape=(r,r)
) -> jnp.ndarray: # scalar, shape=()
    dX1_perp, dY1 = standard_perturbation1
    dX2_perp, dY2 = standard_perturbation2
    t1 = jnp.einsum('ia,ab,ib', dX1_perp, inner_product_helper_matrix,  dX2_perp)
    t2 = np.sum(dY1 * dY2)
    return t1 + t2


@jax.jit
def retract_tangent_vector(
        base: typ.Tuple[
            jnp.ndarray,  # X, shape=(N,r)
            jnp.ndarray,  # Y, shape=(r,M)
        ],
        perturbation: typ.Tuple[
            jnp.ndarray,  # dX, shape=(N,r)
            jnp.ndarray,  # dY, shape=(r,M)
        ],
) -> typ.Tuple[
    jnp.ndarray,  # Q2, shape=(N,r), Q2^T Q2 = I
    jnp.ndarray,  # Y2, shape=(r,M)
]: # retracted_vector based on truncated SVD of dX Y + Y dX. Is left orthogonal, even if base is not
    X, Y = base
    dX, dY = perturbation
    r = X.shape[1]

    bigX = jnp.hstack([X, X, dX])
    bigY = jnp.vstack([Y, dY, Y])
    QX, RX = jnp.linalg.qr(bigX, mode='reduced')
    QYT, RYT = jnp.linalg.qr(bigY.T, mode='reduced')
    U, ss, Vt = jnp.linalg.svd(RX @ RYT.T, full_matrices=False)
    Q = QX @ U[:,:r]
    Y2 = (ss[:r].reshape(-1,1) * Vt[:r,:]) @ (QYT.T)
    retracted_vector = (Q, Y2)
    return retracted_vector


#### Tests

jax.config.update("jax_enable_x64", True) # enable double precision

# Test base_to_full()

N = 17
r = 5
M = 12

X = np.random.randn(N,r)
Y = np.random.randn(r,M)
base = (X, Y)

A = base_to_full(base)
A_true = X @ Y
err_base_to_full = np.linalg.norm(A - A_true) / np.linalg.norm(A_true)
print('err_base_to_full=', err_base_to_full)

# test left_orthogonalize_base()

Q, Y2 = left_orthogonalize_base(base)

A2 = Q @ Y2
err_mult_base = np.linalg.norm(A2 - A_true) / np.linalg.norm(A_true)
print('err_mult_base=', err_mult_base)

err_orth_left_orthogonalize_base = np.linalg.norm(Q.T @ Q - np.eye(r))
print('err_orth_left_orthogonalize_base=', err_orth_left_orthogonalize_base)

# Test tangent_vector_to_full()

dX = np.random.randn(N,r)
dY = np.random.randn(r,M)
perturbation = (dX, dY)

s = 1e-7
v_diff = (base_to_full((X + s*dX, Y + s*dY)) - base_to_full((X, Y))) / s
v = tangent_vector_to_full(base, perturbation)
err_tangent_vector_to_full = np.linalg.norm(v - v_diff) / np.linalg.norm(v_diff)
print('s=', s, ', err_tangent_vector_to_full=', err_tangent_vector_to_full)

# Test add_tangent_vectors()

perturbation1 = perturbation

dX2 = np.random.randn(N,r)
dY2 = np.random.randn(r,M)
perturbation2 = (dX, dY)

perturbation12 = add_tangent_vectors(perturbation1, perturbation2)

v1 = tangent_vector_to_full(base, perturbation1)
v2 = tangent_vector_to_full(base, perturbation2)
v12_true = v1 + v2

v12 = tangent_vector_to_full(base, perturbation12)
err_add_tangent_vectors = np.linalg.norm(v12 - v12_true) / np.linalg.norm(v12_true)
print('err_add_tangent_vectors=', err_add_tangent_vectors)

# Test scale_tangent_vector()

c = np.random.randn()

scaled_perturbation = scale_tangent_vector(perturbation, c)

cv = tangent_vector_to_full(base, scaled_perturbation)
cv_true = c * tangent_vector_to_full(base, perturbation)
err_scale_tangent_vector = np.linalg.norm(cv - cv_true) / np.linalg.norm(cv_true)
print('err_scale_tangent_vector=', err_scale_tangent_vector)

# Test standardize_perturbation()

left_orthogonal_base = left_orthogonalize_base(base)
standard_perturbation = standardize_perturbation(left_orthogonal_base, perturbation)

v_true = tangent_vector_to_full(left_orthogonal_base, perturbation)
v = tangent_vector_to_full(left_orthogonal_base, standard_perturbation)
err_mult_standardize_perturbation = np.linalg.norm(v - v_true) / np.linalg.norm(v)
print('err_mult_standardize_perturbation=', err_mult_standardize_perturbation)

Q, _ = left_orthogonal_base
dX_perp, _ = standard_perturbation

err_perp_standardize_perturbation = np.linalg.norm(Q.T @ dX_perp)
print('err_perp_standardize_perturbation=', err_perp_standardize_perturbation)

# Test inner_product_of_tangent_vectors()

standard_perturbation1 = standardize_perturbation(left_orthogonal_base, perturbation1)
standard_perturbation2 = standardize_perturbation(left_orthogonal_base, perturbation2)

inner_product_helper_matrix = make_inner_product_helper_matrix(left_orthogonal_base)

IP = inner_product_of_tangent_vectors(standard_perturbation1, standard_perturbation2, inner_product_helper_matrix)

v1 = tangent_vector_to_full(left_orthogonal_base, standard_perturbation1)
v2 = tangent_vector_to_full(left_orthogonal_base, standard_perturbation2)
IP_true = np.sum(v1 * v2)

err_inner_product_of_tangent_vectors = np.linalg.norm(IP - IP_true) / np.linalg.norm(IP_true)
print('err_inner_product_of_tangent_vectors=', err_inner_product_of_tangent_vectors)

# Test retract_tangent_vector()

retracted_vector = retract_tangent_vector(base, perturbation)
v = base_to_full(retracted_vector)

U, ss, Vt = np.linalg.svd(base_to_full(base) + tangent_vector_to_full(base, perturbation))
v_true = U[:,:r] @ np.diag(ss[:r]) @ Vt[:r,:]

err_retract_vector = np.linalg.norm(v - v_true) / np.linalg.norm(v_true)
print('err_retract_vector=', err_retract_vector)



#### CG Steihaug

def cg_steihaug(
        hessian_matvec:         typ.Callable[[typ.Any], typ.Any], # u -> H @ u
        gradient:               typ.Any,
        add:            typ.Callable[[typ.Any, typ.Any], typ.Any],                          # (u, v) -> u+v
        scale:          typ.Callable[[typ.Any, typ.Union[float, jnp.ndarray]], typ.Any],    # (u, c) -> c*u
        inner_product:  typ.Callable[[typ.Any, typ.Any], typ.Union[float, jnp.ndarray]],    # (u, v) -> <u,v>
        trust_region_radius:    typ.Union[float, jnp.ndarray],
        rtol:                   typ.Union[float, jnp.ndarray],
        max_iter:               int  = 250,
        display:                bool = True,
        callback:               typ.Callable = None,
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


# Test cg_steihaug

N = 100
cond = 50

U, _, _ = np.linalg.svd(np.random.randn(N, N))
ss = 1 + np.logspace(0, np.log10(cond), N)
H = U @ np.diag(ss) @ U.T

hessian_matvec = lambda x: H @ x
gradient = np.random.randn(N)
add = lambda u,v: u+v
scale = lambda u,c: c*u
inner_product = lambda u,v: np.dot(u,v)

# Check that this generates the same iterates as reference CG from stackoverflow if:
#  - trust region is infinite,
#  - Hessian is positive definite,
#  - tolerance is zero

max_iter=25

pp = []
callback = lambda z: pp.append(z)

p, aux = cg_steihaug(
    hessian_matvec, gradient, add, scale, inner_product,
    trust_region_radius=np.inf, rtol=0.0, max_iter=max_iter, callback=callback,
)


# This basic implementation from stackoverflow agrees with me to machine precision.
# Scipy's CG doesn't seem to agree with me, although the two converge to each other.
# I think this has something to so with where the callback is called in the scipy implementation.

A = H
b = -gradient
x = x0
tol = 0.0

pp_so = []
# https://stackoverflow.com/a/60847526/484944
r = b - A.dot(x)
p = r.copy()
for i in range(max_iter):
    Ap = A.dot(p)
    alpha = np.dot(p, r) / np.dot(p, Ap)
    x = x + alpha * p
    pp_so.append(x)
    r = b - A.dot(x)
    beta = -np.dot(r, Ap) / np.dot(p, Ap)
    p = r + beta * p


err_CG_iterates = np.linalg.norm(np.array(pp) - np.array(pp_so), axis=1) / np.linalg.norm(np.array(pp_so), axis=1)
print('err_CG_iterates=', err_CG_iterates)

# Check that we stop when the tolerance is reached

for rtol in [1e-1, 1e-2, 1e-3]:
    p, aux = cg_steihaug(
        hessian_matvec, gradient, add, scale, inner_product,
        trust_region_radius=np.inf, rtol=rtol, max_iter=N,
    )
    relres = np.linalg.norm(H @ p + gradient) / np.linalg.norm(gradient)
    print('rtol=', rtol, ', relres=', relres)

# Check that we stop when we exit the trust region

norm_p_true = np.linalg.norm(np.linalg.solve(H, -gradient))

for scaling in [0.1, 0.5, 0.9, 0.99]:
    trust_radius = scaling * norm_p_true
    p, aux = cg_steihaug(
        hessian_matvec, gradient, add, scale, inner_product,
        trust_region_radius=trust_radius, rtol=0.0, max_iter=N,
    )
    norm_p = np.linalg.norm(p)
    print('trust_radius=', trust_radius, ', norm_p=', norm_p)

