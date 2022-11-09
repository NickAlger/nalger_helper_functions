import numpy as np
import dolfin as dl
import jax
import jax.numpy as jnp
from scipy.spatial import KDTree
from time import time
import matplotlib.pyplot as plt

from nalger_helper_functions import circle_mesh

mesh = circle_mesh(np.array([0.0, 0.0]), 1.0, 1.0e-2)
# dl.plot(mesh)

Vh = dl.FunctionSpace(mesh, 'CG', 1)
print('Vh.dim()=', Vh.dim())

dof_coords = Vh.tabulate_dof_coordinates()

kdtree = KDTree(dof_coords)

window_centers = list(dof_coords.copy())
window_radii = 0.1 * np.ones(Vh.dim())

initial_ind_groups = list(kdtree.query_ball_point(window_centers, window_radii))

num_nbrs = np.max([len(g) for g in initial_ind_groups])

_, ind_groups = kdtree.query(window_centers, num_nbrs)

dof_coords_groups = dof_coords[ind_groups, :]

centered_coords_groups = dof_coords_groups - dof_coords.reshape((dof_coords.shape[0],1,dof_coords.shape[1]))
rr_groups = jnp.linalg.norm(centered_coords_groups, axis=2)

def make_kernel(a, L):
    return a * jnp.exp(-0.5 * (rr_groups / L) ** 2)

a = 1.0
L = 0.1 / 3.0

t = time()
gg = make_kernel(a, L)
dt_python = time() - t
print('dt_python=', dt_python)

make_kernel_jit = jax.jit(make_kernel)

t = time()
gg2 = make_kernel_jit(a, L)
dt_jit1 = time() - t
print('dt_jit1=', dt_jit1)

t = time()
gg2 = make_kernel_jit(a, L)
dt_jit2 = time() - t
print('dt_jit2=', dt_jit2)

def apply_A(u):
    return jnp.sum(u[ind_groups] * gg, axis=1)

z = np.random.randn(Vh.dim())

t = time()
Az = apply_A(z)
dt_python = time() - t
print('dt_python=', dt_python)

apply_A_jit = jax.jit(apply_A)

t = time()
Az2 = apply_A_jit(z)
dt_jit1 = time() - t
print('dt_jit1=', dt_jit1)

t = time()
Az2 = apply_A_jit(z)
dt_jit2 = time() - t
print('dt_jit2=', dt_jit2)

Az_fct = dl.Function(Vh)
Az_fct.vector()[:] = Az2

dl.plot(Az_fct)