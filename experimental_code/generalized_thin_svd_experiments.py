import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

N = 100
r = 13

Y = np.random.randn(N,r)

Mleft = np.random.randn(N,N)
M = Mleft @ Mleft.T

sqrtM = sla.sqrtm(M)

#

U_true_hat, ss_true, VT_true = np.linalg.svd(sqrtM @ Y, full_matrices=False)

U_true = np.linalg.solve(sqrtM, U_true_hat)

print(np.linalg.norm(U_true.T @ M @ U_true - np.eye(r)))
print(np.linalg.norm(Y - U_true @ np.diag(ss_true) @ VT_true))

#

ss_squared, V = np.linalg.eigh(Y.T @ M @ Y)

idx = ss_squared.argsort()[::-1]
ss_squared = ss_squared[idx]
V = V[:,idx]

ss = np.sqrt(ss_squared)

err_ss = np.linalg.norm(ss - ss_true)
print('err_ss=', err_ss)

plt.matshow(VT_true @ V)
plt.title('VT_true @ V')

U = (Y @ V) / ss.reshape((1,-1))

Y2 = U @ np.diag(ss) @ V.T

err_factorization = np.linalg.norm(Y2 - Y)
print('err_factorization=', err_factorization)

err_inner = np.linalg.norm(U.T @ M @ U - np.eye(r))
print('err_inner=', err_inner)

