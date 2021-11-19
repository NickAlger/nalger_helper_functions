import numpy as np
import hlibpro_python_wrapper as hpro
from time import time
from scipy.spatial import KDTree

hcpp = hpro.hpro_cpp


K = 5
num_neighbors = 11
num_querys = 133

pp = np.random.randn(100,K)
KDT = hcpp.KDTree(np.array(pp.T, order='F'))
KDT.block_size = 1

qq = np.random.randn(num_querys, K)
all_inds, all_dsqq = KDT.query(np.array(qq.T, order='F'), num_neighbors)

err_nearest = 0.0
err_dsqq = 0.0
for ii in range(num_querys):
    q = qq[ii,:]
    inds = all_inds[:,ii]
    dsqq = all_dsqq[:,ii]
    nearest_points = pp[inds,:]

    nearest_inds = np.argsort(np.linalg.norm(pp - q[None,:], axis=1), axis=0)[:num_neighbors]
    nearest_points_true = pp[nearest_inds, :]
    dsqq_true = np.linalg.norm(q[None, :] - nearest_points_true, axis=1) ** 2

    # err = np.linalg.norm(nearest_points - nearest_points_true)
    # print('ii=', ii)
    # print('inds - nearest_inds=', inds - nearest_inds)

    err_nearest += np.linalg.norm(nearest_points - nearest_points_true)
    err_dsqq += np.linalg.norm(dsqq - dsqq_true)

print('err_nearest=', err_nearest)
print('err_dsqq=', err_dsqq)

print('')

#

print('timing:')
K = 3
n_pts = int(1e6)
n_query = int(1e6)
num_neighbors = 10
block_size = 32
print('K=', K, 'n_pts=', n_pts, ', n_query=', n_query, ', num_neighbors=', num_neighbors, ', block_size=', block_size)

pp = np.random.randn(n_pts, K)
pp_T = np.array(pp.T, order='F')

t = time()
KDT = hcpp.KDTree(pp_T)
dt_build = time() - t
print('dt_build=', dt_build)

KDT.block_size = block_size

qq = np.random.randn(n_query, K)
qq_T = np.array(qq.T, order='F')

t = time()
KDT.query(qq_T, 1)
dt_query_one = time() - t
print('dt_query_one=', dt_query_one)

t = time()
KDT.query_multithreaded(qq_T, 1)
dt_query_one_multithreaded = time() - t
print('dt_query_one_multithreaded=', dt_query_one_multithreaded)


t = time()
KDT_scipy = KDTree(pp)
dt_build_scipy = time() - t
print('dt_build_scipy=', dt_build_scipy)

t = time()
KDT_scipy.query(qq)
dt_query_one_scipy = time() - t
print('dt_query_one_scipy=', dt_query_one_scipy)

#

t = time()
KDT.query(qq_T, num_neighbors)
dt_query_many = time() - t
print('dt_query_many=', dt_query_many)

t = time()
KDT.query_multithreaded(qq_T, num_neighbors)
dt_query_multithreaded_many = time() - t
print('dt_query_multithreaded_many=', dt_query_multithreaded_many)

t = time()
KDT_scipy.query(qq, num_neighbors)
dt_query_many_scipy = time() - t
print('dt_query_many_scipy=', dt_query_many_scipy)

# 11/16/21 AFTER THREAD POOLING
# K= 3 n_pts= 1000000 , n_query= 1000000 , num_neighbors= 10 , block_size= 32
# dt_build= 2.2663416862487793
# dt_query_one= 2.196108341217041
# dt_query_one_multithreaded= 0.3906748294830322
# dt_build_scipy= 0.8720104694366455
# dt_query_one_scipy= 3.2524194717407227
# dt_query_many= 6.171878337860107
# dt_query_multithreaded_many= 1.152435541152954
# dt_query_many_scipy= 8.431021451950073

# 11/16/21 BEFORE THREAD POOLING
# K= 3 n_pts= 1000000 , n_query= 1000000 , num_neighbors= 10 , block_size= 32
# dt_build= 1.8702449798583984
# dt_query_one= 2.0514936447143555
# dt_build_scipy= 0.6923236846923828
# dt_query_one_scipy= 3.2761380672454834
# dt_query_many= 5.77819561958313
# dt_query_many_scipy= 7.658422470092773

# Resulting output 11/12/21, after making code N-dimensional, removing one query:
# err_nearest= 0.0
# err_dsqq= 6.068071412950251e-16
#
# timing:
# n_pts= 1000000 , n_query= 1000000 , num_neighbors= 10
# dt_build= 1.735537052154541
# dt_query_one= 2.4149911403656006
# dt_build_scipy= 0.593212366104126
# dt_query_one_scipy= 1.8253419399261475
# dt_query_many= 5.586395740509033
# dt_query_many_scipy= 4.059949636459351

# Resulting output 11/11/21:
#
# one query, one neighbor:
# err_nearest_one_point= 0.0
# err_dsq_one_point= 2.7755575615628914e-17
#
# many querys, many neighbors:
# err_nearest= 0.0
# err_dsqq= 8.930789300493098e-16
#
# timing:
# n_pts= 1000000 , n_query= 1000000 , num_neighbors= 10
# dt_build= 1.491417646408081
# dt_query_one= 1.1302480697631836
# dt_build_scipy= 0.7670538425445557
# dt_query_one_scipy= 2.1079294681549072
# dt_query_many= 3.0853629112243652
# dt_query_many_scipy= 4.660918712615967
