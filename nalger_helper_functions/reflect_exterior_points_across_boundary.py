import numpy as np
import dolfin as dl

from .closest_point_in_mesh import closest_point_in_mesh


def reflect_exterior_points_across_boundary(pp, mesh):
    # pp.shape = (num_points, spatial_dimension)
    pp_projected = closest_point_in_mesh(pp, mesh)
    pp_with_reflection = pp + 2.0 * (pp_projected - pp)
    return pp_with_reflection
