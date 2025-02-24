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

A0 = jnp.einsum('iaj,jk,kal,lm->im', jnp.einsum('ijk,aj->iak', D, W), FH, jnp.einsum('ijk,aj->iak', D, Phi), F)
A = A0 @ A0.T.conj()

A0_true = np.zeros((n,n), dtype=complex)
for ii in range(r):
    A0_true += np.diag(W[ii,:]) @ FH @ np.diag(Phi[ii,:]) @ F

err_make_A0 = np.linalg.norm(A0 - A0_true) / np.linalg.norm(A0_true)
print('err_make_A0=', err_make_A0)

Q0, R0 = np.linalg.qr(W.T, mode='reduced')
Q = Q0.T
R = R0.T

X = jnp.einsum('ab,aj->bj', R, Phi)

B0 = jnp.einsum('iaj,jk,kal,lm->im', jnp.einsum('ijk,aj->iak', D, Q), FH, jnp.einsum('ijk,aj->iak', D, X), F)
B = B0 @ B0.T.conj()

err_orth_W = np.linalg.norm(A0 - B0)
print('err_orth_W=', err_orth_W)

dQA = np.random.randn(*W.shape)
dXA = np.random.randn(*Phi.shape)

tAq = jnp.einsum('iaj,jk,kal,lm->im', jnp.einsum('ijk,aj->iak', D, dQA), FH, jnp.einsum('ijk,aj->iak', D, X), F)
tAx = jnp.einsum('iaj,jk,kal,lm->im', jnp.einsum('ijk,aj->iak', D, Q), FH, jnp.einsum('ijk,aj->iak', D, dXA), F)
tangentA = tAq @ tAq.T.conj() + tAq @ tAx.T.conj() + tAx @ tAq.T.conj() + tAx @ tAx.T.conj()

norm_tangentA = np.linalg.norm(tangentA * tangentA.conj())

C = dQA @ Q.T
dQA_parallel = C @ Q
dQA_perp = dQA - dQA_parallel

dQA2 = dQA_perp
dXA2 = dXA + C.T @ X

tAq2 = jnp.einsum('iaj,jk,kal,lm->im', jnp.einsum('ijk,aj->iak', D, dQA2), FH, jnp.einsum('ijk,aj->iak', D, X), F)
tAx2 = jnp.einsum('iaj,jk,kal,lm->im', jnp.einsum('ijk,aj->iak', D, Q), FH, jnp.einsum('ijk,aj->iak', D, dXA2), F)
tangentA2 = tAq2 @ tAq2.T.conj() + tAq2 @ tAx2.T.conj() + tAx2 @ tAq2.T.conj() + tAx2 @ tAx2.T.conj()

err_perp = np.linalg.norm(tangentA - tangentA2) / np.linalg.norm(tangentA)
print('err_perp=', err_perp)




