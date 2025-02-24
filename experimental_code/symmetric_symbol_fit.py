import numpy as np
import jax
import jax.numpy as jnp
import typing as typ
import functools as ft
import jax.numpy.fft as jfft

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True) # enable double precision

n = 20
r = 5

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

#

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
u1 = tAq2 @ tAq2.T.conj()
u2 = tAq2 @ tAx2.T.conj()
u3 = tAx2 @ tAq2.T.conj()
u4 = tAx2 @ tAx2.T.conj()
tangentA2 = u1 + u2 + u3 + u4

tangentA2 = tAq2 @ tAq2.T.conj() + tAq2 @ tAx2.T.conj() + tAx2 @ tAq2.T.conj() + tAx2 @ tAx2.T.conj()

err_perp = np.linalg.norm(tangentA - tangentA2) / np.linalg.norm(tangentA)
print('err_perp=', err_perp)

#

dQB = np.random.randn(*W.shape)
dXB = np.random.randn(*Phi.shape)

CB = dQB @ Q.T
dQB_parallel = CB @ Q
dQB_perp = dQB - dQB_parallel

dQB2 = dQB_perp
dXB2 = dXB + CB.T @ X

tBq2 = jnp.einsum('iaj,jk,kal,lm->im', jnp.einsum('ijk,aj->iak', D, dQB2), FH, jnp.einsum('ijk,aj->iak', D, X), F)
tBx2 = jnp.einsum('iaj,jk,kal,lm->im', jnp.einsum('ijk,aj->iak', D, Q), FH, jnp.einsum('ijk,aj->iak', D, dXB2), F)
v1 = tBq2 @ tBq2.T.conj()
v2 = tBq2 @ tBx2.T.conj()
v3 = tBx2 @ tBq2.T.conj()
v4 = tBx2 @ tBx2.T.conj()
tangentB2 = v1 + v2 + v3 + v4

# Computing dimension of tangent space

all_tangents = []
k=0
for aa in range(r):
    for ii in range(n):
        dQ = np.zeros((r,n))

        dX = np.zeros(r*n)
        dX[k] = 1.0
        dX = dX.reshape((r,n))

        tq = jnp.einsum('iaj,jk,kal,lm->im', jnp.einsum('ijk,aj->iak', D, dQ), FH, jnp.einsum('ijk,aj->iak', D, X), F)
        tx = jnp.einsum('iaj,jk,kal,lm->im', jnp.einsum('ijk,aj->iak', D, Q), FH, jnp.einsum('ijk,aj->iak', D, dX), F)
        v1 = tq @ tq.T.conj()
        v2 = tq @ tx.T.conj()
        v3 = tx @ tq.T.conj()
        v4 = tx @ tx.T.conj()
        tangent = v1 + v2 + v3 + v4
        # tangent = tq + tx
        all_tangents.append(tangent)

        k += 1

k=0
for aa in range(r):
    for ii in range(n):
        dX = np.zeros((r,n))

        dQ = np.zeros(r*n)
        dQ[k] = 1.0
        dQ = dQ.reshape((r,n))

        tq = jnp.einsum('iaj,jk,kal,lm->im', jnp.einsum('ijk,aj->iak', D, dQ), FH, jnp.einsum('ijk,aj->iak', D, X), F)
        tx = jnp.einsum('iaj,jk,kal,lm->im', jnp.einsum('ijk,aj->iak', D, Q), FH, jnp.einsum('ijk,aj->iak', D, dX), F)
        v1 = tq @ tq.T.conj()
        v2 = tq @ tx.T.conj()
        v3 = tx @ tq.T.conj()
        v4 = tx @ tx.T.conj()
        tangent = v1 + v2 + v3 + v4
        # tangent = tq + tx
        all_tangents.append(tangent)

        k += 1

TTT = np.array(all_tangents)
ss = np.linalg.svd(TTT.reshape((-1, n*n)))[1]

dim_tangent_space = np.sum(ss > 1e-12)
print('dim_tangent_space=', dim_tangent_space)

# (n-r)*r + n*r
print('(n-1)*(r+1) + 1=', (n-1)*(r+1) + 1)

########

if False:
    np.sum((tBq2 @ tBx2.T.conj()) * (tBq2 @ tBq2.T.conj()).conj())

    Xaijb = np.zeros((n,r,r,n), dtype=complex)
    for ii in range(r):
        for jj in range(r):
            Xaijb[:,ii,jj,:] = np.diag(dQA2[ii,:].T.conj() * Q[jj,:])

    #

    IAB_true = np.sum(tangentA2.T.conj() * tangentB2)

    Maijb = np.zeros((n,r,r,n), dtype=complex)
    for ii in range(r):
        for jj in range(r):
            Maijb[:,ii,jj,:] = FH @ np.diag(X[ii,:] * X[jj,:].T.conj()) @ F


    Xaijb = np.zeros((n,r,r,n), dtype=complex)
    for ii in range(r):
        for jj in range(r):
            Xaijb[:,ii,jj,:] = np.diag(dQA2[ii,:].T.conj() * dQB2[jj,:])

    IAB_X = np.sum(Maijb * Xaijb)

    #

    Naijb_Q = np.zeros((n,r,r,n), dtype=complex)
    for ii in range(r):
        for jj in range(r):
            Naijb_Q[:,ii,jj,:] = FH @ np.diag(Q[ii,:] * Q[jj,:].T.conj()) @ F

    Yaijb = np.zeros((n,r,r,n), dtype=complex)
    for ii in range(r):
        for jj in range(r):
            Yaijb[:,ii,jj,:] = np.diag(dQA2[ii,:] * dQB2[jj,:])

    IAB_X = np.sum(Maijb * Xaijb)




    np.sum(np.diag(dXA.conj() * dXB))


    IAB = np.sum(((X.T.conj() @ X)) * (dQA2.T.conj() @ dQB2)) + np.sum((dXA2.conj().T @ dXB2))






