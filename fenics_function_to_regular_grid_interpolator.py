import numpy as np
import dolfin as dl

# https://github.com/NickAlger/helper_functions/
from make_regular_grid import make_regular_grid
from point_is_in_ellipsoid import point_is_in_ellipsoid


class FenicsFunctionToRegularGridInterpolator:
    # https://github.com/NickAlger/helper_functions/blob/master/fenics_function_to_regular_grid_interpolator.ipynb
    def __init__(me, function_space_V, grid_min, grid_max, grid_shape):
        me.V = function_space_V
        me.grid_min = grid_min
        me.grid_max = grid_max
        me.grid_shape = grid_shape

        mesh = me.V.mesh()
        me.d = mesh.geometric_dimension()
        me.bbt = mesh.bounding_box_tree()
        pt_outside_mesh = dl.Point(np.inf * np.ones(mesh.geometric_dimension()))
        me.bad_bbt_entity = me.bbt.compute_first_entity_collision(pt_outside_mesh)

        _, me.all_XX = make_regular_grid(grid_min, grid_max, grid_shape)

    def point_is_in_mesh(me, p_numpy):
        p_fenics = dl.Point(p_numpy)
        if me.bbt.compute_first_entity_collision(p_fenics) == me.bad_bbt_entity:
            return False
        else:
            return True

    def interpolate(me, u_fenics,
                    outside_domain_default_value=0.0, use_extrapolation=False):
        if use_extrapolation:
            u_fenics.set_allow_extrapolation(True)

        U = np.zeros(me.grid_shape)
        for ii in range(np.prod(me.grid_shape)):
            nd_ind = np.unravel_index(ii, me.grid_shape)
            p = np.array([me.all_XX[k][nd_ind] for k in range(me.d)])

            if use_extrapolation:
                U[nd_ind] = u_fenics(dl.Point(p))

            elif me.point_is_in_mesh(p):
                U[nd_ind] = u_fenics(dl.Point(p))

            else:
                U[nd_ind] = outside_domain_default_value

        return U

    def interpolate_within_ellipsoid(me, u_fenics,
                                     mu, Sigma, tau,
                                     outside_domain_default_value=0.0,
                                     inside_domain_default_value=0.0,
                                     use_extrapolation=False):
        if use_extrapolation:
            u_fenics.set_allow_extrapolation(True)

        U = np.zeros(me.grid_shape)
        for ii in range(np.prod(me.grid_shape)):
            nd_ind = np.unravel_index(ii, me.grid_shape)
            p = np.array([me.all_XX[k][nd_ind] for k in range(me.d)])

            if me.point_is_in_mesh(p):
                if point_is_in_ellipsoid(p, mu, Sigma, tau):  # inside mesh, inside ellipsoid
                    U[nd_ind] = u_fenics(dl.Point(p))
                else:  # inside mesh, outside ellipsoid
                    U[nd_ind] = inside_domain_default_value

            elif (not point_is_in_ellipsoid(p, mu, Sigma, tau)):  # outside mesh, outside ellipsoid
                U[nd_ind] = outside_domain_default_value

            else:  # outside mesh, inside ellipsoid
                if use_extrapolation:
                    U[nd_ind] = u_fenics(dl.Point(p))
                else:
                    U[nd_ind] = outside_domain_default_value

        return U