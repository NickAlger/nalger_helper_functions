import numpy as np
import dolfin as dl
from nalger_helper_functions import closest_point_on_simplex

import matplotlib.pyplot as plt

def closest_point_in_mesh(p, mesh):
    if len(p.shape) == 1:
        PP = p[None,:]
    else:
        PP = p
    num_pts, N = PP.shape
    tdim = mesh.topology().dim()
    k = tdim + 1

    VV = np.zeros((num_pts, k, N))
    bbt = mesh.bounding_box_tree()
    for ii in range(N):
        closest_entity, closest_distance = bbt.compute_closest_entity(dl.Point(p))
        closest_cell = mesh.cells()[closest_entity]
        vertices_of_closest_cell = mesh.coordinates()[closest_cell, :]
        VV[ii, :, :] = vertices_of_closest_cell

    projected_PP = closest_point_on_simplex(PP, VV)

    if len(p.shape) == 1:
        projected_PP = projected_PP.reshape(-1)
    return projected_PP





nx=13
ny=9
mesh = dl.UnitSquareMesh(nx,ny)

plt.figure()
dl.plot(mesh)

p = np.array([1.1, 0.05])
plt.plot(p[0], p[1], '*r')


tdim = mesh.topology().dim()
bbt = mesh.bounding_box_tree()
entity, distance = bbt.compute_closest_entity(dl.Point(p))

mesh.geometry()

C = mesh.cells()[entity]
V = mesh.coordinates()[C,:].T
plt.plot(V[0,:], V[1,:], '*b')

print(V)


q_min, c_min, d_min = closest_point_on_triangle(p, V)

print('q_min=', q_min)

q_min2 = closest_point_on_simplex(p, V.T)

print('q_min2=', q_min2)

from time import time


num_pts = 10000
pp = np.random.randn(num_pts, 2)
t = time()
for k in range(num_pts):
    pk = pp[k,:]
    entity, distance = bbt.compute_closest_entity(dl.Point(pk))
    C = mesh.cells()[entity]
    Vk = mesh.coordinates()[C, :].T
    q_mink, c_mink, d_mink = closest_point_on_triangle(pk, Vk)
dt = time() - t
print('dt=', dt)



d=len(p)
# Check edge 0-1
p_hat = p - V[:,0]
V_hat = V[:,1].reshape((d,-1)) - V[:,0].reshape((d,-1))
c_01 = np.linalg.lstsq(V_hat, p_hat, rcond=None)[0]
print('c_01=', c_01)

projected_p01, c01 = project_point_onto_affine_subspace(p, V[:,[0,1]])

p_hat = p - V[:,0]
V_hat = V[:,2].reshape((2,-1)) - V[:,0].reshape((2,-1))
c_02 = np.linalg.lstsq(V_hat, p_hat, rcond=None)[0]
print('c_02=', c_02)

projected_p02, c02 = project_point_onto_affine_subspace(p, V[:,[0,2]])

p_hat = p - V[:,1]
V_hat = V[:,2].reshape((2,-1)) - V[:,1].reshape((2,-1))
c_12 = np.linalg.lstsq(V_hat, p_hat, rcond=None)[0]
print('c_12=', c_12)

projected_p12, c12 = project_point_onto_affine_subspace(p, V[:,[1,2]])

k=2
num_pts = 50000
N=3
PP = np.random.randn(num_pts, N)
VV = np.random.randn(num_pts, k, N)
t = time()
cc = projected_affine_coordinates_vectorized(PP, VV)
dt_vectorized = time() - t
print('dt_vectorized=', dt_vectorized)

t = time()
cc_true = list()
for ii in range(num_pts):
    p = PP[ii,:]
    V = VV[ii, :, :].T
    _, c_true = project_point_onto_affine_subspace(p, V)
    cc_true.append(c_true)
cc_true = np.array(cc_true)
dt_unvectorized = time() - t
print('dt_unvectorized=', dt_unvectorized)

err_vectorized = np.linalg.norm(cc_true - cc)
print('err_vectorized=', err_vectorized)

k=3
num_pts = 50000
N=3
PP = np.random.randn(num_pts, N)
VV = np.random.randn(num_pts, k, N)
t = time()
cc = projected_affine_coordinates_vectorized(PP, VV)
dt_vectorized = time() - t
print('dt_vectorized=', dt_vectorized)

t = time()
cc_true = list()
for ii in range(num_pts):
    _, c_true = project_point_onto_affine_subspace(PP[ii,:], VV[ii, :, :].T)
    cc_true.append(c_true)
cc_true = np.array(cc_true)
dt_unvectorized = time() - t
print('dt_unvectorized=', dt_unvectorized)

err_vectorized = np.linalg.norm(cc_true - cc) / np.linalg.norm(cc_true)
print('err_vectorized=', err_vectorized)

# k, num_pts, N = VV.shape
# VV0 = VV[0, :, :].reshape((1, num_pts, N))
# dVV = VV[1:, :, :].reshape((k - 1, num_pts, N)) - VV0
# dPP = PP.reshape((1, num_pts, N)) - VV0
# RHS = np.sum(dVV * dPP, axis=-1)  # shape = (k-1, num_pts)
# PHI = np.einsum('ixz,jxz->ijx', dVV, dVV).swapaxes(0, 2)  # shape = (num_pts, k-1, k-1)
# iPHI = np.linalg.inv(PHI)  # shape = (num_pts, k-1, k-1)
# cc_rest = np.einsum('pij,jp->pi', iPHI, RHS)  # shape = (num_pts, k-1)
# cc_first = (1. - np.sum(cc_rest, axis=1)).reshape((num_pts, 1))
# affine_coordinates = np.concatenate([cc_first, cc_rest], axis=1)


