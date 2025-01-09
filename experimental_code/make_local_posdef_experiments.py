import numpy as np
import matplotlib.pyplot as plt

# Experimental
# Make local matrix positive semi-definite by:
#  1) breaking the matrix into pieces using a partition of unity of the matrix entries
#  2) making each piece of the matrix positive semi-definite
#  3) adding the semi-definite pieces back together

#### Create spatially varying Gaussian kernel matrix, 'A', in 1D

tt = np.linspace(0, 10, 1000)

V = np.sqrt(1.0 - tt/1.5 + tt**2/3.1 - tt**3/50) # spatially varying volume
mu = tt + 0.1 * np.sin(5*tt) # spatially varying mean
Sigma = (0.5 + (10-tt)/30)**2

plt.figure()
plt.plot(tt, V)
plt.plot(tt, mu)
plt.plot(tt, Sigma)
plt.title('spatially varying moments')
plt.legend(['V', 'mu', 'Sigma'])

A = np.zeros((len(tt), len(tt)))
for ii in range(A.shape[0]):
    pp = tt - mu[ii]
    A[ii,:] = V[ii]*np.exp(-0.5 * pp**2 / Sigma[ii])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for ii in [0,200,400,800,999]:
    plt.plot(tt, A[ii,:])
plt.title('rows of A')

plt.subplot(1,2,2)
for jj in [0,200,400,800,999]:
    plt.plot(tt, A[:,jj])
plt.title('cols of A')

A_sym = 0.5 * (A + A.T)
ee, P = np.linalg.eigh(A_sym)
A_sym_plus = P @ np.diag(ee * (ee > 0)) @ P.T


#### Create partition of unity on matrix space, Psik_hat

# landmark points
p0 = 0.0
p1 = 2.0
p2 = 4.0
p3 = 6.0
p4 = 8.0
p5 = 10.0

# un-normalized partition functions
psi_func = lambda x,y,p: np.exp(-0.5 * np.linalg.norm(np.array([x,y]) - np.array([p,p]))**2/(1.0**2))
Psi0 = np.array([[psi_func(tt[ii], tt[jj], p0) for jj in range(len(tt))] for ii in range(len(tt))])
Psi1 = np.array([[psi_func(tt[ii], tt[jj], p1) for jj in range(len(tt))] for ii in range(len(tt))])
Psi2 = np.array([[psi_func(tt[ii], tt[jj], p2) for jj in range(len(tt))] for ii in range(len(tt))])
Psi3 = np.array([[psi_func(tt[ii], tt[jj], p3) for jj in range(len(tt))] for ii in range(len(tt))])
Psi4 = np.array([[psi_func(tt[ii], tt[jj], p4) for jj in range(len(tt))] for ii in range(len(tt))])
Psi5 = np.array([[psi_func(tt[ii], tt[jj], p5) for jj in range(len(tt))] for ii in range(len(tt))])
all_Psi = [Psi0, Psi1, Psi2, Psi3, Psi4, Psi5]

plt.figure(figsize=(12,5))
for ii, Psi in enumerate(all_Psi):
    plt.subplot(1,len(all_Psi),ii+1)
    plt.imshow(Psi)
    plt.title('Psi'+str(ii))

# partition of unity on matrix space
Psi_sum = Psi0 + Psi1 + Psi2 + Psi3 + Psi4 + Psi5
Psi0_hat = Psi0 / Psi_sum
Psi1_hat = Psi1 / Psi_sum
Psi2_hat = Psi2 / Psi_sum
Psi3_hat = Psi3 / Psi_sum
Psi4_hat = Psi4 / Psi_sum
Psi5_hat = Psi5 / Psi_sum
all_Psi_hat = [Psi0_hat, Psi1_hat, Psi2_hat, Psi3_hat, Psi4_hat, Psi5_hat]

plt.figure(figsize=(12,5))
for ii, Psi_hat in enumerate(all_Psi_hat):
    plt.subplot(1,len(all_Psi_hat),ii+1)
    plt.imshow(Psi_hat)
    plt.title('Psi'+str(ii)+'_hat')

#### Break symmetrized matrix, A_sym, into pieces using partition of unity.

A_sym0 = Psi0_hat * A_sym
A_sym1 = Psi1_hat * A_sym
A_sym2 = Psi2_hat * A_sym
A_sym3 = Psi3_hat * A_sym
A_sym4 = Psi4_hat * A_sym
A_sym5 = Psi5_hat * A_sym
all_A_sym = [A_sym0, A_sym1, A_sym2, A_sym3, A_sym4, A_sym5]

plt.figure(figsize=(12,5))
for ii, A_symk in enumerate(all_A_sym):
    plt.subplot(1,len(all_A_sym),ii+1)
    plt.imshow(A_symk)
    plt.title('A_sym'+str(ii))

#### Make each piece of matrix positive definite and add back together

ee0, P0 = np.linalg.eigh(A_sym0)
ee1, P1 = np.linalg.eigh(A_sym1)
ee2, P2 = np.linalg.eigh(A_sym2)
ee3, P3 = np.linalg.eigh(A_sym3)
ee4, P4 = np.linalg.eigh(A_sym4)
ee5, P5 = np.linalg.eigh(A_sym5)

plt.figure()
plt.plot(ee0)
plt.plot(ee1)
plt.plot(ee2)
plt.plot(ee3)
plt.plot(ee4)
plt.plot(ee5)
plt.title('eigenvalues of components of A_sym')
plt.legend(['A_sym0', 'A_sym1', 'A_sym2', 'A_sym3', 'A_sym4', 'A_sym5'])

A_sym0_plus = P0 @ np.diag(ee0 * (ee0 > 0)) @ P0.T
A_sym1_plus = P1 @ np.diag(ee1 * (ee1 > 0)) @ P1.T
A_sym2_plus = P2 @ np.diag(ee2 * (ee2 > 0)) @ P2.T
A_sym3_plus = P3 @ np.diag(ee3 * (ee3 > 0)) @ P3.T
A_sym4_plus = P4 @ np.diag(ee4 * (ee4 > 0)) @ P4.T
A_sym5_plus = P5 @ np.diag(ee5 * (ee5 > 0)) @ P5.T

A_sym_plus_tilde = A_sym0_plus + A_sym1_plus + A_sym2_plus + A_sym3_plus + A_sym4_plus + A_sym5_plus

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(A_sym_plus)
plt.title('A_sym_plus')
plt.subplot(1,3,2)
plt.imshow(A_sym_plus_tilde)
plt.title('A_sym_plus_tilde')
plt.subplot(1,3,3)
plt.imshow(A_sym_plus_tilde-A_sym_plus)
plt.title('A_sym_plus_tilde-A_sym_plus')

err = np.linalg.norm(A_sym_plus_tilde - A_sym_plus) / np.linalg.norm(A_sym_plus)
print('err=', err)
