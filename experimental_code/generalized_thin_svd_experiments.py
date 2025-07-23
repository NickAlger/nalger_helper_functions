import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


n = 10
m = 7

A = np.random.randn(n, m)

N0 = np.random.randn(n, n)
N = N0 @ N0.T

M0 = np.random.randn(m, m)
M = M0 @ M0.T

k = 4

#

bigA = np.bmat([[np.zeros((n,n)), A], [A.T, np.zeros((m,m))]])
bigM = np.bmat([[N, np.zeros((n,m))], [np.zeros((m,n)), M]])

sigmas_big, UV = sla.eigh(bigA, bigM)

sigmas_big = sigmas_big[::-1]
UV = UV[:,::-1]

k0 = np.minimum(n,m)
U = UV[:n,:k0]
V = UV[n:,:k0]
sigmas = sigmas_big[:k0]

np.linalg.norm(U.T @ N @ U - 0.5*np.eye(U.shape[1])) # what is this 0.5 from?
np.linalg.norm(V.T @ M @ V - 0.5*np.eye(V.shape[1]))


np.linalg.norm(A @ V - N @ U @ np.diag(sigmas))
np.linalg.norm(A.T @ U - M @ V @ np.diag(sigmas))

np.linalg.norm(N @ U @ np.diag(sigmas) @ V.T @ M - 0.5*A) / np.linalg.norm(A)

#

bigA = spla.LinearOperator((n+m, n+m), matvec=apply_bigA)
bigM = spla.LinearOperator((n+m, n+m), matvec=apply_bigM)
bigMinv = spla.LinearOperator((n+m, n+m), matvec=solve_bigM)

lambdas, UV = spla.eigsh(bigA, 7, M=bigM, Minv=bigMinv, which='LM')

U = UV[:n,:]
V = UV[n:,:]

plt.matshow(U.T @ N @ U)
plt.matshow(V.T @ M @ V)

U @ np.diag(lambdas) @ V.T

print(np.linalg.norm(U.T @ N @ U - np.eye(U.shape[1])))
print(V.T @ M @ V - np.eye(V.shape[1]))

U @ np.diag(lambdas) @ V.T - A

#

def alternative_svd(A, N, M, k):
    lambdas1, P = sla.eigh(A.T @ N @ A)
    inds = np.argsort(lambdas1)[::-1]
    Pk = P[:,inds]
    B = (A @ Pk)
    lambdas, Q = sla.eigh(B.T @ N @ B)
    sigmas = np.sqrt(lambdas)
    U = (B @ Q) / sigmas.reshape((1, -1))
    V = P @ Q
    return U, ss, V.T


#

lambdas1, P = sla.eigh(A.T @ N @ A, M)

sort_inds1 = np.argsort(lambdas1)[::-1]
lambdas1 = lambdas1[sort_inds1]
P = P[:, sort_inds1]

Pk = P[:, :k]
B = (A @ M @ Pk)
# lambdas2, Qk = sla.eigh(B.T @ N @ B, Pk.T @ M @ Pk)
lambdas2, Qk = sla.eigh(B.T @ N @ B)

sort_inds2 = np.argsort(lambdas2)[::-1]
lambdas2 = lambdas2[sort_inds2]
Qk = Qk[:, sort_inds2]

# sigmas = np.sqrt(lambdas2)
# Uk = (B @ Qk) / sigmas.reshape((1, -1))
# Vk = Pk @ Qk
sigmas = np.sqrt(lambdas1[:k])

Uk = (A @ Pk) / sigmas.reshape((1, -1))
Sigmak = np.diag(sigmas)

# Vk = (Pk.T @ Pk @ Sigmak @ Pk.T).T
Vk = Pk


Uk @ Sigmak @ Pk.T - A

# X = Uk.T @ A @ Pk
#
# Uk @ X @ Pk.T - A

np.linalg.norm(Uk.T @ N @ Uk - np.eye(Uk.shape[1]))
np.linalg.norm(Pk.T @ M @ Pk - np.eye(Pk.shape[1]))
np.linalg.norm(Uk @ Sigmak @ Pk.T - A) / np.linalg.norm(A)

#

lambda1, P = sla.eigh(A.T @ N @ A)

B = (A @ P)

print(np.linalg.norm(B @ P.T - A)) # A = B V1.T
print(np.linalg.norm(P.T @ P - np.eye(P.shape[1]))) # V1.T M V1 = I

lambda2, Q = sla.eigh(B.T @ M @ B)

print(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))) # V2.T V2 = I

U1 = (B @ Q) / np.sqrt(lambda2).reshape((1,-1))

B2 = U1 @ np.diag(np.sqrt(lambda2)) @ Q.T

print(np.linalg.norm(B2 - B))
print(np.linalg.norm(U1.T @ N @ U1 - np.eye(U1.shape[1])))

V3 = P @ Q
A2 = U1 @ np.diag(np.sqrt(lambda2)) @ V3.T
np.linalg.norm(A2 - A)

#



U = (B @ M @ V2) / np.sqrt(lambda2).reshape((1,-1))
V = V2 @ V1

Sigma = np.diag(np.sqrt(lambda2))

A2 = U @ Sigma @ V.T

err_factorization = np.linalg.norm(A2 - A) / np.linalg.norm(A)
print('err_factorization=', err_factorization)

err_N = np.linalg.norm(U.T @ N @ U - np.eye(U.shape[1]))
print('err_N=', err_N)

err_M = np.linalg.norm(V.T @ M @ V - np.eye(V.shape[1]))
print('err_M=', err_M)

#

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

