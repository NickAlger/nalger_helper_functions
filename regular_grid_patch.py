import numpy as np
import dolfin as dl

# https://github.com/NickAlger/helper_functions/
from point_is_in_box import point_is_in_box
from pointcloud_nearest_neighbor import pointcloud_nearest_neighbor
from fenics_function_to_regular_grid_interpolator import FenicsFunctionToRegularGridInterpolator
from grid_interpolator import GridInterpolator
from conforming_box import conforming_box
from make_regular_grid import make_regular_grid


class RegularGridPatch:
    # https://github.com/NickAlger/helper_functions/blob/master/regular_grid_patch.ipynb
    def __init__(me, function_space_V, approximate_box_min, approximate_box_max,
                 anchor_point_p, grid_density_multiplier=1.0):
        me.V = function_space_V
        me.p = anchor_point_p
        me.grid_density_multiplier = grid_density_multiplier

        dof_coords = me.V.tabulate_dof_coordinates()

        approximate_box_mask = point_is_in_box(dof_coords,
                                               approximate_box_min,
                                               approximate_box_max)
        points_in_approximate_box = dof_coords[approximate_box_mask, :]

        h0 = np.min(pointcloud_nearest_neighbor(points_in_approximate_box,
                                                return_min_distances=True)[1])

        me.h = h0 / grid_density_multiplier

        me.box_min, me.box_max, me.grid_shape = conforming_box(approximate_box_min,
                                                               approximate_box_max,
                                                               me.p, me.h)

        me.box_mask = point_is_in_box(dof_coords, me.box_min, me.box_max)
        me.inds_of_points_in_box = np.argwhere(me.box_mask).reshape(-1)
        me.points_in_box = dof_coords[me.box_mask, :]

        me.function_to_grid_object = FenicsFunctionToRegularGridInterpolator(me.V,
                                                                             me.box_min,
                                                                             me.box_max,
                                                                             me.grid_shape)
        me.grid_to_points_object = GridInterpolator(me.box_min,
                                                    me.box_max,
                                                    me.grid_shape)

        me.all_xx, me.all_XX = make_regular_grid(me.box_min, me.box_max, me.grid_shape)

    def function_to_grid(me, u_fenics_function,
                         ellipse_mu=None, ellipse_Sigma=None, ellipse_tau=None,
                         use_extrapolation=False):
        U_array = me.function_to_grid_object.interpolate(u_fenics_function,
                                                         mu=ellipse_mu,
                                                         Sigma=ellipse_Sigma,
                                                         tau=ellipse_tau,
                                                         use_extrapolation=use_extrapolation)
        return U_array

    def grid_to_function(me, U_array):
        u = dl.Function(me.V)
        u.vector()[me.inds_of_points_in_box] = me.grid_to_points_object.interpolate(U_array,
                                                                                    me.points_in_box)
        return u
