import numpy as np
import matplotlib.pyplot as plt


# Make spatially varying gaussian kernel matrix

tt = np.linspace(0, 10, 1000)

V = np.sqrt(1.0 - tt/1.5 + tt**2/3.1 - tt**3/50) # spatially varying volume
mu = tt + 0.1 * np.sin(5*tt) # spatially varying mean
Sigma = 9*(0.5 + (10-tt)/30)**2

plt.figure()
plt.plot(tt, V)
plt.plot(tt, mu)
plt.plot(tt, Sigma)
plt.title('spatially varying moments')
plt.legend(['V', 'mu', 'Sigma'])

def get_A_block(rr, cc):
    block = np.zeros((len(rr), len(cc)))
    for i in range(len(rr)):
        r = rr[i]
        pp = tt[list(cc)] - mu[r]
        block[i,:] = V[r]*np.exp(-0.5 * pp**2 / Sigma[r])
    return block

all_inds = np.arange(len(tt), dtype=int)
A = get_A_block(all_inds, all_inds)

row_inds = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
col_inds = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]

A_rows = A[row_inds,:]
A_cols = A[:,col_inds]

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for ii in range(len(row_inds)):
    plt.plot(tt, A_rows[ii,:])
plt.title('rows of A')

plt.subplot(1,2,2)
for jj in range(len(col_inds)):
    plt.plot(tt, A_cols[:,jj])
plt.title('cols of A')

#


_, _, Vt_row = np.linalg.svd(A_rows, full_matrices=False)
U_col, _, _ = np.linalg.svd(A_cols, full_matrices=False)

plt.figure(figsize=(25,10))
for ii in range(Vt_row.shape[0]):
    plt.subplot(2, Vt_row.shape[0]+1, ii+1)
    plt.plot(tt, Vt_row[ii,:])
plt.title('Row sample basis')

for ii in range(Vt_row.shape[0]):
    plt.subplot(2, Vt_row.shape[0]+1, Vt_row.shape[0]+1 + ii+1)
    plt.plot(tt, U_col[:,ii])
plt.title('Col sample basis')

# Add noise

noise_level = 2e-1

noise = np.random.randn(*A.shape)
noise = noise * noise_level * np.linalg.norm(A) / np.linalg.norm(noise)

A_noisy = A + noise

#

A_rows_noisy = A_noisy[row_inds,:]
A_cols_noisy = A_noisy[:,col_inds]

_, _, Vt_row_noisy = np.linalg.svd(A_rows_noisy, full_matrices=False)
U_col_noisy, _, _ = np.linalg.svd(A_cols_noisy, full_matrices=False)

plt.figure(figsize=(25,10))
for ii in range(Vt_row_noisy.shape[0]):
    plt.subplot(2, Vt_row_noisy.shape[0]+1, ii+1)
    plt.plot(tt, Vt_row_noisy[ii,:])
plt.title('Noisy row sample basis')

for ii in range(Vt_row_noisy.shape[0]):
    plt.subplot(2, Vt_row_noisy.shape[0]+1, Vt_row_noisy.shape[0]+1 + ii+1)
    plt.plot(tt, U_col_noisy[:,ii])
plt.title('Noisy col sample basis')

# # Make Laplacian matrix

L0 = np.diag(-np.ones(len(tt)-1),-1) + np.diag(2*np.ones(len(tt)),0) + np.diag(-np.ones(len(tt)-1),1) # neumann laplacian

L1 = L0.copy() # Dirichlet Laplacian
L1[0,0] = 1
L1[-1,-1] = 1

UL, ssL, VtL = np.linalg.svd(L1)
ssL2 = ssL.copy()
ssL2[-1] = ssL2[-2]
L2 = UL @ np.diag(ssL2) @ VtL # neumann Laplacian without zero eigenvalue

# Generalized SVD.
# See: https://toussile.quarto.pub/singular-value-decomposition/generalized_svd.html

A_cols_noisy_tilde = np.linalg.solve(L2, A_cols_noisy) # L2 @ A_cols_noisy #np.linalg.solve(L2, A_cols_noisy)
A_rows_noisy_tilde = np.linalg.solve(L2.T, A_rows_noisy.T).T # A_rows_noisy @ L2

U_col_noisy_tilde, _, _ = np.linalg.svd(A_cols_noisy_tilde, full_matrices=False)
_, _, Vt_row_noisy_tilde = np.linalg.svd(A_rows_noisy_tilde, full_matrices=False)

U_cols_smoothed = U_col_noisy_tilde # L2 @ U_col_noisy_tilde # np.linalg.solve(L2, U_col_noisy_tilde) #L2 @ U_col_noisy_tilde
Vt_rows_smoothed = Vt_row_noisy_tilde # Vt_row_noisy_tilde @ L2 #np.linalg.solve(L2.T, Vt_row_noisy_tilde.T).T


plt.figure(figsize=(25,10))
for ii in range(Vt_row_noisy.shape[0]):
    plt.subplot(2, Vt_row_noisy.shape[0]+1, ii+1)
    plt.plot(tt, U_cols_smoothed[:,ii])
plt.title('Smoothed noisy row sample basis')

for ii in range(Vt_row_noisy.shape[0]):
    plt.subplot(2, Vt_row_noisy.shape[0]+1, Vt_row_noisy.shape[0]+1 + ii+1)
    plt.plot(tt, Vt_rows_smoothed[ii,:])
plt.title('Smoothed noisy col sample basis')

#
r = 5

Z = np.linalg.pinv(U_col_noisy[row_inds[:r],:r]) @ Vt_row_noisy[:r,:]

A2 = U_col_noisy[:,:r] @ Z


Z_smooth = np.linalg.solve(U_cols_smoothed[row_inds,:r], Vt_row[:r,:])


A2_smooth = U_cols_smoothed @ Z_smooth

np.linalg.norm(A2 - A) / np.linalg.norm(A)

#

_,ss_center,_ = np.linalg.svd(U_col[row_inds,:])
_,ss_center_smooth,_ = np.linalg.svd(U_cols_smoothed[row_inds,:])






if False:
    # Plot singular vectors

    U, ss, Vt = np.linalg.svd(A)

    nplot = 15
    plt.figure(figsize=(25, 10))
    for ii in range(nplot):
        plt.subplot(2, nplot, ii + 1)
        plt.plot(tt, U[:, ii])
    plt.title('Top left singular vectors')

    for ii in range(nplot):
        plt.subplot(2, nplot, nplot + ii + 1)
        plt.plot(tt, Vt[ii, :])
    plt.title('Top right singular vectors')

    # Plot noisy singular vectors

    U_noisy, ss_noisy, Vt_noisy = np.linalg.svd(A_noisy)

    nplot = 15
    plt.figure(figsize=(25, 10))
    for ii in range(nplot):
        plt.subplot(2, nplot, ii + 1)
        plt.plot(tt, U_noisy[:, ii])
    plt.title('Top noisy left singular vectors')

    for ii in range(nplot):
        plt.subplot(2, nplot, nplot + ii + 1)
        plt.plot(tt, Vt_noisy[ii, :])
    plt.title('Top noisy right singular vectors')
