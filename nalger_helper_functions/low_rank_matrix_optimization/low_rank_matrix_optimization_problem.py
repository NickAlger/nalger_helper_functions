import numpy as np
import jax.numpy as jnp
import jax
import typing as typ

import nalger_helper_functions.tree_linalg as tla


__all__ = [
    'forward_map',
    'forward_map_jvp',
    'forward_map_vjp',
    'regularization',
    'regularization_gradient',
    'regularization_hessian_matvec',
    'spd_sqrtm',
    'objective',
    'gradient',
    'gauss_newton_hessian_matvec',
    # 'tangent_space_misfit',
]


Param = typ.Tuple[
    jnp.ndarray,  # X, shape=(N,r)
    jnp.ndarray,  # Y, shape=(r,M)
]
ParamTangent = typ.Tuple[
    jnp.ndarray,  # X, shape=(N,r)
    jnp.ndarray,  # Y, shape=(r,M)
]
ParamCoTangent = typ.Tuple[
    jnp.ndarray,  # X, shape=(N,r)
    jnp.ndarray,  # Y, shape=(r,M)
]

Inputs = typ.Tuple[
    jnp.ndarray, # Omega, shape=(M,k)
    jnp.ndarray, # Omega_r, shape=(k_r,N)
]

LossAux = typ.Tuple[
    jnp.ndarray, # relerrs, shape=(k,)
    jnp.ndarray, # relerrs_r, shape=(k_r)

]
Outputs = typ.Tuple[
    jnp.ndarray,  # Z, shape=(N,k)
    jnp.ndarray,  # Z_r, shape=(k_r,M)
]
OutputsTangent = typ.Tuple[
    jnp.ndarray,  # Z, shape=(N,k)
    jnp.ndarray,  # Z_r, shape=(k_r,M)
]
OutputsCoTangent = typ.Tuple[
    jnp.ndarray,  # Z, shape=(N,k)
    jnp.ndarray,  # Z_r, shape=(k_r,M)
]

Scalar  = typ.TypeVar('Scalar')

@jax.jit
def forward_map(
        base:   Param,
        inputs: Inputs,
) -> Outputs:
    X, Y = base
    Omega, Omega_r = inputs
    Z = X @ (Y @ Omega)
    Z_r = Omega_r @ (X @ Y)
    outputs = Z, Z_r
    return outputs


@jax.jit
def forward_map_jvp(
        base:           Param,
        inputs:         Inputs,
        perturbation:   ParamTangent,
) -> OutputsTangent:
    '''Returns:
    Z, Z_r = lim s->0 forward_map((X + s*dX, Y + s*dY), Omega) - forward_map((X, Y), Omega) / s
    '''
    X, Y = base
    dX, dY = perturbation
    Omega, Omega_r = inputs
    Z = dX @ (Y @ Omega) + X @ (dY @ Omega)
    Z_r = (Omega_r @ dX) @ Y + (Omega_r @ X) @ dY
    return Z, Z_r


@jax.jit
def forward_map_vjp(
        base:   Param,
        inputs: Inputs,
        ZZ:     OutputsCoTangent,
) -> ParamCoTangent:
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
        base: Param,
        apply_R: typ.Callable[[Param], Scalar],
):
    '''
    t1: 1/2 * ||ML @ X @ Y||^2
    t2: 1/2 * ||X @ Y @ MR||^2
    t3: 1/2 * ||ML @ X @ Y @ MR||^2
    '''
    X, Y = base
    Xhat, Yhat = apply_R(base)
    # t1 = 0.5 * np.sum((Xhat @ Y)**2) # <-- slow
    t1 = 1e2 * 0.5 * np.sum((Xhat.T @ Xhat) * (Y @ Y.T)) # <-- fast
    t2 = 0.5 * np.sum((X.T @ X) * (Yhat @ Yhat.T))
    t3 = 0.5 * np.sum((Xhat.T @ Xhat) * (Yhat @ Yhat.T))
    # return t1
    return t1 + t2
    # return t1 + t2 + t3
    # return t3 # double sided


regularization_gradient = jax.grad(regularization)

def regularization_hessian_matvec(
        base: Param,
        perturbation: ParamTangent,
        apply_R: typ.Callable[[ParamTangent], ParamCoTangent],
) -> ParamCoTangent:
    g_func = lambda b: regularization_gradient(b, apply_R)
    return jax.jvp(g_func, (base,), (perturbation,))[1]

#

@jax.jit
def loss(
        Y: Outputs,
        Y_true: Outputs
) -> typ.Tuple[
    Scalar,
    LossAux,
]:
    y, y_r = Y
    yt, yt_r = Y_true
    rsq_num = jnp.sum((y - yt)**2, axis=0) ** 2
    rsq_den = jnp.sum(yt**2, axis=0) ** 2
    rsq_num_r = jnp.sum((y_r - yt_r)**2, axis=1) ** 2
    rsq_den_r = jnp.sum(yt_r**2, axis=1) ** 2
    relerrs = rsq_num / rsq_den
    relerrs_r = rsq_num_r / rsq_den_r

    J = 0.5 * jnp.sum(rsq_num) + 0.5 * jnp.sum(rsq_num_r)
    return J, (relerrs, relerrs_r)


_loss_grad_helper = jax.grad(loss, argnums=0, has_aux=True)
loss_grad = jax.jit(lambda *args, **kwargs: _loss_grad_helper(*args, **kwargs)[0])


@jax.jit
def loss_gnhvp(y: Outputs, y_true: Outputs, dy: OutputsTangent) -> OutputsCoTangent:
    g_func = lambda z: loss_grad(z, y_true)
    return jax.jvp(g_func, (y,), (dy,))[1]


@jax.jit
def misfit(
        m:              Param,
        x:              Inputs,
        y_true:         Outputs, # true observations
) -> typ.Tuple[
    Scalar, # misfit Jd
    typ.Tuple[
        Outputs,  # predicted outputs y based on the model m
        LossAux,
    ]
]:
    y  = forward_map(m, x)
    J, J_aux = loss(y, y_true)
    return J, (y, J_aux)


@jax.jit
def misfit_gradient(
        m:      Param,
        x:      Inputs,
        y:      Outputs,
        y_true: Outputs,
) -> ParamCoTangent: # gradient g
    z: OutputsCoTangent = loss_grad(y, y_true)
    g, fvjp_aux = forward_map_vjp(m, x,  z)
    return g, fvjp_aux


@jax.jit
def misfit_gauss_newton_hessian_matvec(
        dm: ParamTangent,
        m:  Param,
        x:  Inputs,
        y:  Outputs,
        y_true:  Outputs,
) -> ParamCoTangent: # H @ dm, Gauss-Newton Hessian vector product
    dy = forward_map_jvp(m, x, dm)
    dy2 = loss_gnhvp(y, y_true, dy)
    Hdm = forward_map_vjp(m, x, dy2)
    return Hdm


def objective(
        m:      Param,
        x:      Inputs,
        y_true: Outputs,
        a_reg: Scalar,  # regularization parameter
        apply_R: typ.Callable[[ParamTangent], ParamCoTangent],
) -> typ.Tuple[
    Scalar, # J, objective
    typ.Tuple[
        Scalar, # Jd, misfit component
        Scalar, # Jr, regularization component
        Outputs, # outputs y
        LossAux,
    ]
]:
    Jd, (y, Jd_aux) = misfit(m, x, y_true)
    Jr0 = regularization(m, apply_R)
    Jr = tla.scale(Jr0, a_reg)
    J = tla.add(Jd, Jr)
    return J, (Jd, Jr, y, Jd_aux)


def gradient(
        m: Param,
        x: Inputs,
        y:      Outputs,
        y_true: Outputs,
        a_reg: Scalar,
        apply_R: typ.Callable[[ParamTangent], ParamCoTangent],
) -> typ.Tuple[
    ParamCoTangent, # gradient g
    typ.Tuple[
        ParamCoTangent, # gd, misfit component
        ParamCoTangent, # gr, # regularization component
    ]
]:
    gd = misfit_gradient(m, x, y, y_true)
    gr0 = regularization_gradient(m, apply_R)
    gr = tla.scale(gr0, a_reg)
    g = tla.add(gd, gr)
    return g, (gd, gr)


def gauss_newton_hessian_matvec(
        dm: ParamTangent,
        m: Param,
        x: Inputs,
        y: Outputs,
        y_true: Outputs,
        a_reg: Scalar,
        apply_R: typ.Callable[[ParamTangent], ParamCoTangent],
) -> typ.Tuple[
    ParamCoTangent, # H @ dm, Gauss-Newton Hessian matvec
    typ.Tuple[
        ParamCoTangent, # Hd @ dm, misfit component
        ParamCoTangent, # Hr @ dm, misfit component
    ]
]:
    Hd_dm = misfit_gauss_newton_hessian_matvec(dm, m, x, y, y_true)
    Hr_dm0 = regularization_hessian_matvec(m, dm, apply_R)
    Hr_dm = tla.scale(Hr_dm0, a_reg)
    H_dm = tla.add(Hd_dm, Hr_dm)
    return H_dm, (Hd_dm, Hr_dm)



@jax.jit
def spd_sqrtm(A):
    ee, P = jnp.linalg.eigh(A)
    return P @ (jnp.sqrt(jnp.abs(ee)).reshape((-1,1)) * P.T)
