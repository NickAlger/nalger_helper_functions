import numpy as np
import dolfin as dl
from scipy.spatial import cKDTree
from nalger_helper_functions import point_is_in_box,\
    FenicsFunctionToRegularGridInterpolator, \
    grid_interpolate, conforming_box, make_regular_grid


class RegularGridPatch:
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/regular_grid_patch.ipynb
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

        T = cKDTree(points_in_approximate_box)
        dd, _ = T.query(points_in_approximate_box, k=2)
        h0 = np.min(dd[:, 1])

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
        u.vector()[me.inds_of_points_in_box] = grid_interpolate(me.box_min, me.box_max, U_array, me.points_in_box)
        return u
