import numpy as np
import jax.numpy as jnp
import jax

from nalger_helper_functions.least_squares_framework import *

jax.config.update("jax_enable_x64", True) # enable double precision

def forward_map(m, x):
    u0 = jnp.sin(m[0] * x[1]) + x[2]
    u1 = jnp.cos(x[0]) + x[1] * jnp.exp(m[1]) / x[3]
    u2 = 3.5 + x[0] + m[0] * m[1] / x[1]
    return jnp.array([u0, u1, u2]), None

Jac_func = jax.jacobian(forward_map, argnums=0, has_aux=True)

def forward_map_jvp(m, x, dm, f_aux):
    F, _ = Jac_func(m, x)
    return F @ dm, None

def forward_map_vjp(m, x, du, f_aux):
    F, _ = Jac_func(m, x)
    return F.T @ du, None

x = np.random.randn(4)
y_true = np.random.randn(3)
a_reg = np.random.rand()

J_func = lambda m: objective(m, x, y_true, forward_map, a_reg)
g_func = lambda m, f_aux, y: gradient(m, x, f_aux, y, y_true, a_reg, forward_map_vjp)
Hgn_matvec_func = lambda dm, m, f_aux, y: gauss_newton_hessian_matvec(dm, m, x, f_aux, y, forward_map_jvp, forward_map_vjp, a_reg)


m0 = np.random.randn(2)
J0, (Jd0, Jr0, f_aux0, y0, Jd_aux0) = J_func(m0)
g0, (gd0, gr0, fvjp_aux0) = g_func(m0, f_aux0, y0)

dm = np.random.randn(2)
dJ = np.dot(g0, dm)
dJd = np.dot(gd0, dm)
dJr = np.dot(gr0, dm)
H_dm0, (Hd_dm0, Hr_dm0, fjvp_aux0, fvjp_aux0) = Hgn_matvec_func(dm, m0, f_aux0, y0)
dg = H_dm0

s = 1e-7

m1 = m0 + s * dm
J1, (Jd1, Jr1, f_aux1, y1, Jd_aux1) = J_func(m1)
g1, (gd1, gr1, fvjp_aux1) = g_func(m1, f_aux1, y1)

dJ_diff = (J1 - J0) / s
dJd_diff = (Jd1 - Jd0) / s
dJr_diff = (Jr1 - Jr0) / s
err_g = np.abs(dJ - dJ_diff) / np.abs(dJ_diff)
err_gd = np.abs(dJd - dJd_diff) / np.abs(dJd_diff)
err_gr = np.abs(dJr - dJr_diff) / np.abs(dJr_diff)
print('s=', s, ', err_g=', err_g)
print('s=', s, ', err_gd=', err_gd)
print('s=', s, ', err_gr=', err_gr)

# Finite check Gauss-Newton Hessian where residual is zero

y_true = forward_map(m0, x)[0]

J_func = lambda m: objective(m, x, y_true, forward_map, a_reg)
g_func = lambda m, f_aux, y: gradient(m, x, f_aux, y, y_true, a_reg, forward_map_vjp)
Hgn_matvec_func = lambda dm, m, f_aux, y: gauss_newton_hessian_matvec(dm, m, x, f_aux, y, forward_map_jvp, forward_map_vjp, a_reg)

J0, (Jd0, Jr0, f_aux0, y0, Jd_aux0) = J_func(m0)
g0, (gd0, gr0, fvjp_aux0) = g_func(m0, f_aux0, y0)

dm = np.random.randn(2)
dg, (dgd, dgr, fjvp_aux0, fvjp_aux0) = Hgn_matvec_func(dm, m0, f_aux0, y0)

s = 1e-7

m1 = m0 + s * dm
J1, (Jd1, Jr1, f_aux1, y1, Jd_aux1) = J_func(m1)
g1, (gd1, gr1, fvjp_aux1) = g_func(m1, f_aux1, y1)

dg_diff = (g1 - g0) / s
dgd_diff = (gd1 - gd0) / s
dgr_diff = (gr1 - gr0) / s
err_H = np.linalg.norm(dg - dg_diff) / np.linalg.norm(dg_diff)
err_Hd = np.linalg.norm(dgd - dgd_diff) / np.linalg.norm(dgd_diff)
err_Hr = np.linalg.norm(dgr - dgr_diff) / np.linalg.norm(dgr_diff)
print('s=', s, ', err_H=', err_H)
print('s=', s, ', err_Hd=', err_Hd)
print('s=', s, ', err_Hr=', err_Hr)

