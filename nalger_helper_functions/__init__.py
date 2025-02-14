# Ordering matters! Must import dependencies before dependents

try:
    from .dtype_max import dtype_max
except ImportError:
    print('dtype_max not loaded')

try:
    from .ind2sub_batches import ind2sub_batches, sub2ind_batches
except ImportError:
    print('ind2sub_batches, sub2ind_batches not loaded')

try:
    from .build_dense_matrix_from_matvecs import build_dense_matrix_from_matvecs
except ImportError:
    print('build_dense_matrix_from_matvecs not loaded')

try:
    from .invert_dictionary import invert_dictionary
except ImportError:
    print('invert_dictionary not loaded')

try:
    from .vec2fct import vec2fct
except ImportError:
    print('vec2fct not loaded')

try:
    from .make_dense_lu_solver import make_dense_lu_solver
except ImportError:
    print('make_dense_lu_solver not loaded')

try:
    from .box_mesh_lexsort import box_mesh_lexsort
except ImportError:
    print('box_mesh_lexsort not loaded')

try:
    from .box_mesh_nd import box_mesh_nd
except ImportError:
    print('box_mesh_nd not loaded')

try:
    from .circle_mesh import circle_mesh
except ImportError:
    print('circle_mesh not loaded')

try:
    from .conforming_box import conforming_box
except ImportError:
    print('conforming_box not loaded')

try:
    from .conforming_grid_convolution import conforming_grid_convolution
except ImportError:
    print('conforming_grid_convolution not loaded')

try:
    from .csr_fenics2scipy import csr_fenics2scipy
except ImportError:
    print('csr_fenics2scipy not loaded')

try:
    from .csr_scipy2fenics import csr_scipy2fenics
except ImportError:
    print('csr_scipy2fenics not loaded')

try:
    from .ellipsoid_bounding_box import ellipsoid_bounding_box
except ImportError:
    print('ellipsoid_bounding_box not loaded')

try:
    from .function_support_box import function_support_box
except ImportError:
    print('function_support_box')

try:
    from .interactive_impulse_response_plot import interactive_impulse_response_plot
except ImportError:
    print('fenics_interactive_impulse_response_plot not loaded')

try:
    from .grid_interpolate import grid_interpolate
except ImportError:
    print('grid_interpolate not loaded')

try:
    from .make_regular_grid import make_regular_grid
except ImportError:
    print('make_regular_grid not loaded')

try:
    from .neumann_poisson_solver import NeumannPoissonSolver
except ImportError:
    print('NeumannPoissonSolver not loaded')

try:
    from .plot_ellipse import plot_ellipse
except ImportError:
    print('plot_ellipse not loaded')

try:
    from .plot_rectangle import plot_rectangle
except ImportError:
    print('plot_rectangle not loaded')

try:
    from .point_is_in_box import point_is_in_box
except ImportError:
    print('point_is_in_box not loaded')

try:
    from .point_is_in_ellipsoid import point_is_in_ellipsoid
except ImportError:
    print('point_is_in_ellipsoid not loaded')

try:
    from .rational_inverse_square_root_weights_and_poles import rational_inverse_square_root_weights_and_poles
except ImportError:
    print('rational_inverse_square_root_weights_and_poles not loaded')

try:
    from .unit_speed_derivatives import unit_speed_derivatives
except ImportError:
    print('unit_speed_derivatives not loaded')

try:
    from .poisson_squared_interpolation import PoissonSquaredInterpolation
except ImportError:
    print('PoissonSquaredInterpolation not loaded')

try:
    from .fenics_function_to_regular_grid_interpolator import FenicsFunctionToRegularGridInterpolator
except ImportError:
    print('FenicsFunctionToRegularGridInterpolator not loaded')

try:
    from .regular_grid_patch import RegularGridPatch
except ImportError:
    print('RegularGridPatch not loaded')

try:
    from .combine_grid_functions import combine_grid_functions
except ImportError:
    print('combine_grid_functions not loaded')

try:
    from .factorized import factorized
except ImportError:
    print('factorized not loaded')

try:
    from .least_squares_derivative_matrix import least_squares_directional_derivative_matrix
except ImportError:
    print('least_squares_directional_derivative_matrix not loaded')

try:
    from .box_function import BoxFunction, boxconv, boxinner, boxnorm, boxintegrate, convolution_square_root, box_functions_are_conforming, ellipsoid_characteristic_function
except ImportError:
    print('BoxFunction, boxconv, boxinner, boxnorm, boxintegrate, convolution_square_root, box_functions_are_conforming not loaded')

try:
    from .pointwise_observation_matrix import pointwise_observation_matrix, points_inside_mesh, dofs_that_contribute_to_function_at_points
except ImportError:
    print('pointwise_observation_matrix, points_inside_mesh, dofs_that_contribute_to_function_at_points not loaded')

try:
    from .fit_sparse_matrix import fit_sparse_matrix
except ImportError:
    print('fit_sparse_matrix not loaded')

try:
    from .multilinear_interpolation_matrix import multilinear_interpolation_matrix, unit_box_corners
except ImportError:
    print('multilinear_interpolation_matrix, unit_box_corners not loaded')

try:
    from .shortest_distance_between_points_in_box import shortest_distance_between_points_in_box
except ImportError:
    print('shortest_distance_between_points_in_box not loaded')

try:
    from .load_image_into_fenics import load_image_into_fenics
except ImportError:
    print('load_image_into_fenics not loaded')

try:
    from .custom_cg import custom_cg
except ImportError:
    print('custom_cg not loaded')

try:
    from .dlfct2array import dlfct2array, array2dlfct
except ImportError:
    print('dlfct2array, array2dlfct not loaded')

try:
    from .function_space_tensor2scalar import function_space_tensor2scalar, function_space_scalar2tensor
except ImportError:
    print('function_space_tensor2scalar, function_space_scalar2tensor not loaded')

try:
    from .ellipsoid_intersection_test import ellipsoid_intersection_test, ellipsoid_intersection_test_helper, ellipsoid_K_function
except ImportError:
    print('ellipsoid_intersection_test, ellipsoid_intersection_test_helper, ellipsoid_K_function not loaded')

try:
    from .vector_polynomial import VectorPolynomial, VectorRationalFunction, \
        taylor_vector_polynomial, pade_vector_rational_function, \
        vector_polynomial_interpolation, polynomial_coeffs_from_roots, vector_floater_hormann_rational_interpolation
except ImportError:
    print('VectorPolynomial, VectorRationalFunction, make_taylor_vector_polynomial, pade_vector_rational_function, '
          + 'vector_polynomial_interpolation, polynomial_coeffs_from_roots, '
          + 'vector_floater_hormann_rational_interpolation not loaded')

try:
    from .make_mass_matrix import make_mass_matrix
except ImportError:
    print('make_mass_matrix not loaded')

try:
    from .make_fenics_amg_solver import make_fenics_amg_solver
except ImportError:
    print('make_fenics_amg_solver not loaded')

try:
    from .powerset import powerset
except ImportError:
    print('powerset not loaded')

try:
    from .project_point_onto_affine_subspace import project_point_onto_affine_subspace
except ImportError:
    print('project_point_onto_affine_subspace not loaded')

try:
    from .closest_point_on_simplex import closest_point_on_simplex
except ImportError:
    print('closest_point_on_simplex not loaded')

try:
    from .closest_point_in_mesh import closest_point_in_mesh
except ImportError:
    print('closest_point_in_mesh not loaded')

try:
    from .eval_fenics_function_on_regular_grid import eval_fenics_function_on_regular_grid
    from .eval_fenics_function_on_regular_grid import eval_fenics_function_on_regular_grid_using_boxmesh
    from .eval_fenics_function_on_regular_grid import eval_fenics_function_on_regular_grid_using_direct_evaluation
    from .eval_fenics_function_on_regular_grid import eval_fenics_function_on_regular_grid_using_pointwise_observation_operator
except ImportError:
    print('val_fenics_function_on_regular_grid not loaded')

try:
    from .reflect_exterior_points_across_boundary import reflect_exterior_points_across_boundary
except ImportError:
    print('reflect_exterior_points_across_boundary not loaded')

try:
    from .nalger_helper_functions_cpp import KDTree
except ImportError:
    print('KDTree not loaded')

try:
    from .nalger_helper_functions_cpp import AABBTree
except ImportError:
    print('AABBTree not loaded')

try:
    from .nalger_helper_functions_cpp import SimplexMesh
except ImportError:
    print('SimplexMesh not loaded')

try:
    from .nalger_helper_functions_cpp import geometric_sort
except ImportError:
    print('geometric_sort not loaded')

try:
    from .nalger_helper_functions_cpp import brent_minimize
except ImportError:
    print('brent_minimize not loaded')

try:
    from .nalger_helper_functions_cpp import ellipsoids_intersect
except ImportError:
    print('ellipsoids_intersect not loaded')

try:
    from .nalger_helper_functions_cpp import EllipsoidBatchPicker
except ImportError:
    print('EllipsoidBatchPicker not loaded')

try:
    from .estimate_column_errors_randomized import estimate_column_errors_randomized
except ImportError:
    print('estimate_column_errors_randomized not loaded')

try:
    from .generalized_leapfrog_integrator import generalized_leapfrog_integrator
except ImportError:
    print('generalized_leapfrog_integrator not loaded')

try:
    from .function_space_enhanced import FunctionSpaceEnhanced
except ImportError:
    print('FunctionSpaceEnhanced not loaded')

try:
    from .threshold_crossings import threshold_crossings
except ImportError:
    print('threshold_crossings not loaded')

try:
    from .lbfgs import lbfgs, LbfgsResult, LbfgsInverseHessianApproximation, LbfgsTerminationReason
except ImportError:
    print('lbfgs not loaded')

try:
    from .cross_approximation import aca_full, aca_partial, recompress_low_rank_approximation
except ImportError:
    print('aca_full, aca_partial not loaded')

try:
    from .low_rank_approximation import low_rank_approximation
except ImportError:
    print('low_rank_approximation not loaded')

try:
    import low_rank_matrix_manifold as lrmm
except ImportError:
    print('low_rank_matrix_manifold not loaded')

try:
    from .cg_steihaug import cg_steihaug
except ImportError:
    print('cg_steihaug not loaded')

try:
    from .trust_region_optimize import trust_region_optimize
except ImportError:
    print('trust_region_optimize not loaded')