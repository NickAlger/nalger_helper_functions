import numpy as np
import dolfin as dl

from nalger_helper_functions import box_mesh_nd, box_mesh_lexsort, points_inside_mesh, make_regular_grid, pointwise_observation_matrix


def eval_fenics_function_on_regular_grid_using_pointwise_observation_operator(f, box_min, box_max, grid_shape,
                                                                              outside_mesh_fill_value=None):
    _, _, gridpoints = make_regular_grid(box_min, box_max, grid_shape, return_gridpoints=True)
    B, inside_domain = pointwise_observation_matrix(gridpoints, f.function_space(), return_inside_mesh_mask=True)
    F_vec = B * f.vector()[:]

    if not (outside_mesh_fill_value is None):
        outside_domain = np.logical_not(inside_domain)
        F_vec[outside_domain] = outside_mesh_fill_value

    F = F_vec.reshape(grid_shape)
    return F


def eval_fenics_function_on_regular_grid_using_direct_evaluation(f, box_min, box_max, grid_shape,
                                                                 outside_mesh_fill_value=None):
    V = f.function_space()
    mesh = V.mesh()
    if outside_mesh_fill_value is None:
        lingrids = make_regular_grid(box_min, box_max, grid_shape, return_meshgrid=False)
    else:
        lingrids, _, gridpoints = make_regular_grid(box_min, box_max, grid_shape, return_gridpoints=True)

    f.set_allow_extrapolation(True)

    F = np.zeros(grid_shape)
    for ind in np.ndindex(grid_shape):
        p = np.array([lingrids[k][ind[k]] for k in range(len(ind))])
        F[ind] = f(dl.Point(p))

    if not (outside_mesh_fill_value is None):
        inside_domain = points_inside_mesh(gridpoints, mesh).reshape(grid_shape)
        outside_domain = np.logical_not(inside_domain)
        F[outside_domain] = outside_mesh_fill_value

    return F


def eval_fenics_function_on_regular_grid_using_boxmesh(f, box_min, box_max, grid_shape,
                                                       outside_mesh_fill_value=None):
    grid_mesh = box_mesh_nd(box_min, box_max, grid_shape)
    V_grid = dl.FunctionSpace(grid_mesh, 'CG', 1)
    lexsort_inds = box_mesh_lexsort(V_grid)

    f.set_allow_extrapolation(True)
    f_grid = dl.Function(V_grid)

    dl.LagrangeInterpolator.interpolate(f_grid, f)

    F = f_grid.vector()[lexsort_inds].reshape(grid_shape)

    if not (outside_mesh_fill_value is None):
        lexsorted_grid_coords = V_grid.tabulate_dof_coordinates()[lexsort_inds, :]
        inside_domain = points_inside_mesh(lexsorted_grid_coords, mesh).reshape(grid_shape)
        outside_domain = np.logical_not(inside_domain)
        F[outside_domain] = outside_mesh_fill_value

    return F


eval_fenics_function_on_regular_grid = eval_fenics_function_on_regular_grid_using_boxmesh


from time import time
from nalger_helper_functions import circle_mesh


mesh = circle_mesh(np.zeros(2), 1.0, 0.05)
V = dl.FunctionSpace(mesh, 'CG', 2)

u = dl.Function(V)
u.vector()[:] = np.random.randn(V.dim())

box_min = np.array([-0.3, -0.4])
box_max = np.array([0.8, 1.0])
grid_shape = (51,47)

outside_mesh_fill_value = -1.2

t = time()
U1 = eval_fenics_function_on_regular_grid_using_boxmesh(u, box_min, box_max, grid_shape,
                                                        outside_mesh_fill_value=outside_mesh_fill_value)
dt_boxmesh = time() - t
print('dt_boxmesh=', dt_boxmesh)

t = time()
U2 = eval_fenics_function_on_regular_grid_using_direct_evaluation(u, box_min, box_max, grid_shape,
                                                                  outside_mesh_fill_value=outside_mesh_fill_value)
dt_direct = time() - t
print('dt_direct=', dt_direct)

t = time()
U3 = eval_fenics_function_on_regular_grid_using_pointwise_observation_operator(u, box_min, box_max, grid_shape,
                                                                               outside_mesh_fill_value=outside_mesh_fill_value)
dt_pointwise_obs = time() - t
print('dt_pointwise_obs=', dt_pointwise_obs)

err_boxmesh_vs_direct = np.linalg.norm(U1-U2) / np.linalg.norm(U1)
print('err_boxmesh_vs_direct=', err_boxmesh_vs_direct)

err_pointwiseobs_vs_direct = np.linalg.norm(U1-U3) / np.linalg.norm(U1)
print('err_pointwiseobs_vs_direct=', err_pointwiseobs_vs_direct)
