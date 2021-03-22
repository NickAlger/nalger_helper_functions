import numpy as np
import dolfin as dl
from nalger_helper_functions import point_is_in_ellipsoid, box_mesh_nd, box_mesh_lexsort


class FenicsFunctionToRegularGridInterpolator:
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/blob/master/jupyter_notebooks/fenics_function_to_regular_grid_interpolator.ipynb
    def __init__(me, function_space_V, grid_min, grid_max, grid_shape):
        me.V = function_space_V
        me.grid_min = grid_min
        me.grid_max = grid_max
        me.grid_shape = grid_shape

        me.N_grid = np.prod(me.grid_shape)

        mesh = me.V.mesh()
        me.d = mesh.geometric_dimension()

        me.grid_mesh = box_mesh_nd(me.grid_min, me.grid_max, me.grid_shape)
        me.V_grid = dl.FunctionSpace(me.grid_mesh, 'CG', 1)
        me.lexsort_inds = box_mesh_lexsort(me.V_grid)

        # Get masks for gridpoints inside and outside the domain
        me.bbt = mesh.bounding_box_tree()
        pt_outside_mesh = dl.Point(np.inf * np.ones(mesh.geometric_dimension()))
        me.bad_bbt_entity = me.bbt.compute_first_entity_collision(pt_outside_mesh)

        me.lexsorted_grid_coords = me.V_grid.tabulate_dof_coordinates()[me.lexsort_inds, :]

        me.inside_domain = np.zeros(me.N_grid, dtype=bool)
        for ii in range(me.N_grid):
            p = me.lexsorted_grid_coords[ii,:]
            if me.point_is_in_mesh(p):
                me.inside_domain[ii] = True
        me.inside_domain = me.inside_domain.reshape(me.grid_shape)
        me.outside_domain = np.logical_not(me.inside_domain)


    def point_is_in_mesh(me, p_numpy):
        p_fenics = dl.Point(p_numpy)
        if me.bbt.compute_first_entity_collision(p_fenics) == me.bad_bbt_entity:
            return False
        else:
            return True

    def interpolate(me, u_fenics,
                    outside_domain_default_value=0.0,
                    inside_domain_default_value=0.0,
                    use_extrapolation=False,
                    mu=None, Sigma=None, tau=None):
        u_fenics.set_allow_extrapolation(True)
        u_fenics_grid = dl.Function(me.V_grid)

        dl.LagrangeInterpolator.interpolate(u_fenics_grid, u_fenics)

        if mu is not None:
            inside_ellipse = point_is_in_ellipsoid(me.lexsorted_grid_coords, mu, Sigma, tau)
        else:
            inside_ellipse = np.ones(me.N_grid, dtype=bool)
        inside_ellipse = inside_ellipse.reshape(me.grid_shape)
        outside_ellipse = np.logical_not(inside_ellipse)

        U = u_fenics_grid.vector()[me.lexsort_inds].reshape(me.grid_shape)

        U[np.logical_and(me.inside_domain, outside_ellipse)] = inside_domain_default_value
        U[np.logical_and(me.outside_domain, outside_ellipse)] = outside_domain_default_value
        if (not use_extrapolation):
            U[np.logical_and(me.outside_domain, inside_ellipse)] = outside_domain_default_value

        return U

