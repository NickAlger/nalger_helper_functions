import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from jax import config
config.update("jax_enable_x64", True)

N = 500
num_sample_rows = 20
num_sample_cols = num_sample_rows

U_true = np.linalg.svd(np.random.randn(N,N))[0]
ss_true = 1.0 / np.power(np.arange(N)+1.0, 2.0)

A = U_true @ np.diag(ss_true) @ U_true.T

col_inds = jnp.array(np.random.permutation(N)[:num_sample_cols])
row_inds = jnp.array(np.random.permutation(N)[:num_sample_rows])

C = jnp.array(A[:,col_inds])
U = jnp.array(C[row_inds,:])
R = jnp.array(A[row_inds,:])

A_cur = C @ np.linalg.pinv(U) @ R

err_cur = np.linalg.norm(A_cur - A) / np.linalg.norm(A)
print('err_cur=', err_cur)

def misfit_func(X, Y):
    JL = 0.5 * jnp.sum((X @ Y[:,col_inds] - C)**2)
    JR = 0.5 * jnp.sum((X[row_inds,:] @ Y - R)**2)
    return JL + JR

def left_regularization_func(X, omega, sigmas):
    return omega**2 * jnp.sum((jnp.einsum('ia,a->ia', X, sigmas[0] * (0.0 + 1.0 / sigmas)))**2)

def right_regularization_func(Y, omega, sigmas):
    return omega**2 * jnp.sum((jnp.einsum('a,aj->aj', sigmas[0] * (0.0 + 1.0 / sigmas), Y))**2)

def objective_X_func(X, Y, omega, sigmas):
    return misfit_func(X, Y) + left_regularization_func(X, omega, sigmas)

def objective_Y_func(X, Y, omega, sigmas):
    return misfit_func(X, Y) + right_regularization_func(Y, omega, sigmas)

grad_X_func = jax.grad(objective_X_func, argnums=0)
grad_Y_func = jax.grad(objective_Y_func, argnums=1)
hess_X_func = jax.hessian(objective_X_func, argnums=0)
hess_Y_func = jax.hessian(objective_Y_func, argnums=1)

def lr_svd(X, Y):
    Q, Z = jnp.linalg.qr(X, mode='reduced')
    U0, sigmas, Vt = jnp.linalg.svd(Z @ Y, 0)
    U = Q @ U0
    return U, sigmas, Vt

def solve4(M_4tensor, b_2tensor):
    n1, n2, n3, n4 = M_4tensor.shape
    return jnp.linalg.solve(M_4tensor.reshape((n1*n2, n3*n4)), b_2tensor.reshape(n1*n2)).reshape((n3, n4))


rank = 5
X = jnp.array(np.random.randn(N, rank))
Y = jnp.array(np.random.randn(rank, N))

omega_schedule = list(np.logspace(-6,0,30)[::-1])
num_sweeps = 1

for omega in omega_schedule:
    omega = jnp.array(omega)
    for sweep in range(num_sweeps):
        U, ss, Vt = lr_svd(X, Y)
        X = U
        Y = jnp.einsum('a,aj->aj', ss, Vt)
        ss_filtered = jnp.maximum(ss, 1e-6 * ss[0])

        err = np.linalg.norm(X @ Y - A) / np.linalg.norm(A)
        Jd = misfit_func(X, Y)
        J = objective_Y_func(X, Y, omega, ss_filtered)
        g = grad_Y_func(X, Y, omega, ss_filtered)
        H = hess_Y_func(X, Y, omega, ss_filtered)
        print('omega=', omega, ', Jd=', Jd, ', err=', err, ', ss=', ss)

        Y = Y - solve4(H, g)

        U, ss, Vt = lr_svd(X, Y)
        X = jnp.einsum('ia,a->ia', U, ss)
        Y = Vt
        ss_filtered = jnp.maximum(ss, 1e-6 * ss[0])

        err = np.linalg.norm(X @ Y - A) / np.linalg.norm(A)
        Jd = misfit_func(X, Y)
        J = objective_X_func(X, Y, omega, ss_filtered)
        g = grad_X_func(X, Y, omega, ss_filtered)
        H = hess_X_func(X, Y, omega, ss_filtered)
        print('omega=', omega, ', Jd=', Jd, ', err=', err, ', ss=', ss)

        X = X - solve4(H, g)

