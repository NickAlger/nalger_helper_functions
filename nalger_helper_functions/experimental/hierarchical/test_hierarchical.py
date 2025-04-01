import numpy as np
import itertools
import matplotlib.pyplot as plt

import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.special as spsc
import scipy.optimize as sopt

import nalger_helper_functions.experimental.hierarchical.hierarchical as hi

from jax.config import config
config.update("jax_enable_x64", True)


#

N = 50
d = 2
pointcloud = np.random.randn(N, d)
perm_e2i, perm_i2e, ct = hi.build_cluster_tree_geometric(pointcloud, min_cluster_size=4)

hi.visualize_cluster_tree_2d(perm_e2i, ct, pointcloud)

#

d = 2
N = 1000
M = 1500
row_pointcloud = np.random.rand(N, d)
col_pointcloud = np.random.rand(M, d)
row_perm_e2i, row_perm_i2e, row_ct2 = hi.build_cluster_tree_geometric(row_pointcloud, min_cluster_size=16)
col_perm_e2i, col_perm_i2e, col_ct2 = hi.build_cluster_tree_geometric(col_pointcloud, min_cluster_size=16)

hi.visualize_cluster_tree_2d(row_perm_e2i, row_ct2, row_pointcloud)
hi.visualize_cluster_tree_2d(col_perm_e2i, col_ct2, col_pointcloud)

#

bct = hi.build_block_cluster_tree(
    row_ct2, col_ct2,
    row_pointcloud, col_pointcloud,
    row_perm_e2i, col_perm_e2i,
    admissibility_eta=0.5
)

hi.visualize_block_cluster_tree(bct)

#

bct = hi.build_block_cluster_tree(
    col_ct2, col_ct2,
    col_pointcloud, col_pointcloud,
    col_perm_e2i, col_perm_e2i,
    admissibility_eta=0.5
)

hi.visualize_block_cluster_tree(bct)

#

bct = hi.build_block_cluster_tree(
    row_ct2, row_ct2,
    row_pointcloud, row_pointcloud,
    row_perm_e2i, row_perm_e2i,
    admissibility_eta=1e-10
)

hi.visualize_block_cluster_tree(bct)

#

pp1 = np.random.randn(1000,2) + np.array([[-1.5, 0.0]])
pp2 = np.random.randn(1000,2) + np.array([[1.5, 0.0]])
pp3 = np.random.randn(1000,2) + np.array([[0.0, 1.5]])
pp = np.vstack([pp1, pp2, pp3])
plt.scatter(pp1[:,0], pp1[:,1])
plt.scatter(pp2[:,0], pp2[:,1])
plt.scatter(pp3[:,0], pp3[:,1])

k=3

Q1 = hi.polynomial_pointcloud_basis(pp1, k)
Q2 = hi.polynomial_pointcloud_basis(pp2, k)
Q3 = hi.polynomial_pointcloud_basis(pp3, k)
Q = hi.polynomial_pointcloud_basis(pp, k)

transfer_matrices = hi.make_transfer_matrices(Q, [Q1, Q2, Q3])

Q2 = np.vstack([Qi @ T for Qi, T  in zip([Q1, Q2, Q3], transfer_matrices)])

err_transfer = np.linalg.norm(Q2 - Q) / np.linalg.norm(Q)
print('err_transfer=', err_transfer)

if False:
    Q1_x = Q[:pp1.shape[0]]
    Q2_x = Q[pp1.shape[0]:]

    np.linalg.norm(Q1 @ (Q1.T @ Q1_x) - Q1_x)

    pp = np.random.randn(5, 2)
    mu = np.mean(pp, axis=0)
    print(mu)
    S = 0.5 * (np.max(pp, axis=0) - np.min(pp, axis=0))
    print(S)

    #

    pointcloud = np.random.randn(6, 2)
    center = 0.5 * (np.max(pointcloud, axis=0) + np.min(pointcloud, axis=0))
    scale = 0.5 * (np.max(pointcloud, axis=0) - np.min(pointcloud, axis=0))
    xx = (pointcloud - center) / scale
    print(xx)

    #

    pointcloud = np.random.randn(1000, 2)
    polynomial_order_k = 3

    N, d = pointcloud.shape
    assert (N > 1)
    center = 0.5 * (np.max(pointcloud, axis=0) + np.min(pointcloud, axis=0))
    scale = 0.5 * (np.max(pointcloud, axis=0) - np.min(pointcloud, axis=0))
    xx = (pointcloud - center) / scale
    polys = list()
    for exponents in itertools.product(range(polynomial_order_k + 1), repeat=d):
        if np.sum(exponents) <= polynomial_order_k:
            kk = np.array(exponents).reshape((1, d))
            poly = np.prod(np.power(xx, kk), axis=1)
            polys.append(poly)
    B = np.array(polys).T

    print('B.shape=', B.shape)

    #

    for ii in range(B.shape[1]):
        plt.figure()
        plt.scatter(pointcloud[:, 0], pointcloud[:, 1], c=B[:, ii])
        plt.colorbar()

    #

    np.linalg.svd(B)[1]
    Q, _, _ = np.linalg.svd(B, 0)
    Q.shape

    for ii in range(Q.shape[1]):
        plt.figure()
        plt.scatter(pointcloud[:, 0], pointcloud[:, 1], c=Q[:, ii])
        plt.colorbar()

    #

