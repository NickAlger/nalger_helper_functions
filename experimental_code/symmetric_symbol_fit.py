import numpy as np
import jax
import jax.numpy as jnp
import typing as typ
import functools as ft
import jax.numpy.fft as jfft

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True) # enable double precision

n = 20
r = 4

F = jfft.fft(np.eye(n,n), norm='ortho')
# F = jfft.fft(np.eye(n,n), norm='forward')
# F = np.linalg.qr(np.random.randn(n,n))[0]

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

#

if False:
    DW = jnp.einsum('ijk,aj->iak', D, W)
    DPhi2 = jnp.einsum('ijk,aj->iak', D, Phi*Phi.conj())

    C = jnp.einsum('iaj,jk,kal,lm,man->in', DW, FH, DPhi2, F, DW.conj())

    np.linalg.norm(A - C)

    #

    D4 = np.einsum('ijk,klm->ijlm', D, D)

    D4_true = np.zeros((n,n,n,n))
    for ii in range(n):
        D4_true[ii,ii,ii,ii] = 1.0

    np.linalg.norm(D4 - D4_true)

    #

    A = np.random.randn(n,n)
    iA = np.linalg.inv(A)

    # W = np.einsum('ia,dj,abcd->ijbc', iA, A, D4).reshape((400,400))
    # W = np.einsum('ia,bj,abcd->icjd', F.T.conj(), F, D4).reshape((400,400))
    W = np.einsum('ia,dj,abcd->ijbc', F.T.conj(), F, D4).reshape((400,400))
    W2 = np.einsum('ia,dj,abcd->ijcb', F.T.conj(), F, D4).reshape((400,400))
    np.linalg.norm(W - W2)

    plt.matshow(W.real)
    plt.matshow(W.imag)

    Wt = np.einsum('ia,dj,abcd->ijbc', F.T.conj(), F, D4)
    plt.matshow(Wt[2,:,3,:].real)

    ya = np.random.randn(n)
    yb = np.random.randn(n)

    z1 = np.einsum('ijbc,b,c', Wt, ya, yb)
    plt.matshow(z1.real)

    ya2 = ya.copy()
    yb2 = yb.copy()
    ya2[1] = 0

    z2 = np.einsum('ijbc,b,c', Wt, ya2, yb2)
    plt.matshow(z2.real)

    plt.matshow((W.T @ np.random.randn(n*n)).reshape(n,n).real)

#

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
        # tangent = v4
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
        # tangent = v4
        # tangent = tq + tx
        all_tangents.append(tangent)

        k += 1

TTT = np.array(all_tangents)
ss = np.linalg.svd(TTT.reshape((-1, n*n)))[1]

dim_tangent_space = np.sum(ss > 1e-12)
print('dim_tangent_space=', dim_tangent_space)

# (n-r)*r + n*r
print('(n-1)*(r+1) + 1=', (n-1)*(r+1) + 1)
print('nr+n-r=', n*r + n - r)

#### Nonpositive tangent space

# Computing dimension of tangent space

n = 20
r = 4

W = np.random.randn(r,n)
Phi = np.random.randn(r,n)

all_tangents = []
k=0
for aa in range(r):
    for ii in range(n):
        dW = np.zeros((r,n))

        dPhi = np.zeros(r*n)
        dPhi[k] = 1.0
        dPhi = dPhi.reshape((r,n))

        v1 = np.sum([np.diag(W[ii,:]) @ FH @ np.diag(Phi[ii,:]) @ F @ np.diag(dW[ii,:]) for ii in range(r)], axis=0)
        v2 = np.sum([np.diag(dW[ii,:]) @ FH @ np.diag(Phi[ii,:]) @ F @ np.diag(W[ii,:]) for ii in range(r)], axis=0)
        v3 = np.sum([np.diag(W[ii, :]) @ FH @ np.diag(dPhi[ii, :]) @ F @ np.diag(W[ii, :]) for ii in range(r)], axis=0)

        tangent = v1 + v2 + v3
        all_tangents.append(tangent)

        k += 1

k=0
for aa in range(r):
    for ii in range(n):
        dPhi = np.zeros((r,n))

        dW = np.zeros(r*n)
        dW[k] = 1.0
        dW = dW.reshape((r,n))

        v1 = np.sum([np.diag(W[ii,:]) @ FH @ np.diag(Phi[ii,:]) @ F @ np.diag(dW[ii,:]) for ii in range(r)], axis=0)
        v2 = np.sum([np.diag(dW[ii,:]) @ FH @ np.diag(Phi[ii,:]) @ F @ np.diag(W[ii,:]) for ii in range(r)], axis=0)
        v3 = np.sum([np.diag(W[ii, :]) @ FH @ np.diag(dPhi[ii, :]) @ F @ np.diag(W[ii, :]) for ii in range(r)], axis=0)

        tangent = v1 + v2 + v3
        all_tangents.append(tangent)

        k += 1

TTT = np.array(all_tangents)
ss = np.linalg.svd(TTT.reshape((-1, n*n)))[1]

dim_tangent_space = np.sum(ss > 1e-12)
print('dim_tangent_space=', dim_tangent_space)

# (n-r)*r + n*r
print('2*n*r-r=', 2*n*r-r)
# print('nr+n-r=', n*r + n - r)

#

n = 20
r = 4

W0 = np.random.randn(r,n)
Phi0 = np.random.randn(r,n)

W_rowsums = np.sum(W0, axis=1).reshape((-1,1))

W = W0 / W_rowsums
Phi = Phi0 * W_rowsums**2

A_true  = np.sum([np.diag(W0[ii,:]) @ FH @ np.diag(Phi0[ii,:]) @ F @ np.diag(W0[ii,:]) for ii in range(r)], axis=0)
A       = np.sum([np.diag(W[ii,:]) @ FH @ np.diag(Phi[ii,:]) @ F @ np.diag(W[ii,:]) for ii in range(r)], axis=0)

err_row_rescaling = np.linalg.norm(A - A_true) / np.linalg.norm(A_true)
print('err_row_rescaling=', err_row_rescaling)

all_tangents = []
k=0
for aa in range(r):
    for ii in range(n):
        dW = np.zeros((r,n))

        dPhi = np.zeros(r*n)
        dPhi[k] = 1.0
        dPhi = dPhi.reshape((r,n))

        v1 = np.sum([np.diag(W[ii,:]) @ FH @ np.diag(Phi[ii,:]) @ F @ np.diag(dW[ii,:]) for ii in range(r)], axis=0)
        v2 = np.sum([np.diag(dW[ii,:]) @ FH @ np.diag(Phi[ii,:]) @ F @ np.diag(W[ii,:]) for ii in range(r)], axis=0)
        v3 = np.sum([np.diag(W[ii, :]) @ FH @ np.diag(dPhi[ii, :]) @ F @ np.diag(W[ii, :]) for ii in range(r)], axis=0)

        tangent = v1 + v2 + v3
        all_tangents.append(tangent)

        k += 1


n_basis = np.linalg.svd(np.eye(n) - 1/n)[0]

k=0
for aa in range(r):
    for ii in range(n-1):
        dPhi = np.zeros((r,n))

        # dW0 = np.zeros(r*n)
        # dW0[k] = 1.0
        # dW0 = dW0.reshape((r,n))
        #
        # dW_rowsums = np.sum(dW0, axis=1).reshape((-1, 1))
        #
        # dW = dW0 - dW_rowsums / n

        dW = np.zeros((r,n))
        dW[aa,:] = n_basis[:,ii]

        v1 = np.sum([np.diag(W[ii,:]) @ FH @ np.diag(Phi[ii,:]) @ F @ np.diag(dW[ii,:]) for ii in range(r)], axis=0)
        v2 = np.sum([np.diag(dW[ii,:]) @ FH @ np.diag(Phi[ii,:]) @ F @ np.diag(W[ii,:]) for ii in range(r)], axis=0)
        v3 = np.sum([np.diag(W[ii, :]) @ FH @ np.diag(dPhi[ii, :]) @ F @ np.diag(W[ii, :]) for ii in range(r)], axis=0)

        tangent = v1 + v2 + v3
        all_tangents.append(tangent)

        k += 1

TTT = np.array(all_tangents)
ss = np.linalg.svd(TTT.reshape((-1, n*n)))[1]

dim_tangent_space = np.sum(ss > 1e-12)
print('dim_tangent_space=', dim_tangent_space)

print('2*n*r-r=', 2*n*r-r)


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






