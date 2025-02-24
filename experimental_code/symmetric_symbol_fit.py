import numpy as np
import jax
import jax.numpy as jnp
import typing as typ
import functools as ft
import jax.numpy.fft as jfft

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True) # enable double precision

n = 20
r = 6

F = jfft.fft(np.eye(n,n), norm='ortho')

err_fft_h = np.linalg.norm(F.T.conj() @ F - np.eye(n))
print('err_fft_h=', err_fft_h)


D = np.zeros((n,n,n))
for ii in range(n):
    D[ii,ii,ii] = 1.0
D = jnp.array(D)

z = np.random.randn(n)
M1 = jnp.einsum('iaj,a->ij', D, z)
M2 = jnp.diag(z)
err_diag_maker = np.linalg.norm(M1 - M2)
print('err_diag_maker=', err_diag_maker)

FH = F.T.conj()

W = np.random.randn(r,n)
Phi = np.random.randn(r,n)

S1 = jnp.einsum('ijk,aj->iak', D, W)
S2 = jnp.einsum('ijk,aj->iak', D, Phi)

A0 = jnp.einsum('iaj,jk,kal,lm->im', S1, FH, S2, F)
A = A0 @ A0.T

Q0, R0 = np.linalg.qr(W.T, mode='reduced')
Q = Q0.T
R = R0.T

X = jnp.einsum('ab,aj->bj', R, Phi)

T1 = jnp.einsum('ijk,aj->iak', D, Q)
T2 = jnp.einsum('ijk,aj->iak', D, X)

B0 = jnp.einsum('iaj,jk,kal,lm->im', T1, FH, T2, F)
B = B0 @ B0.T

err_orth_W = np.linalg.norm(A0 - B0)
print('err_orth_W=', err_orth_W)