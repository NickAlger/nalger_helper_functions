# Ordering matters! Must import dependencies before dependents

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
    from .box_function import BoxFunction, boxconv, boxinner, boxnorm, boxintegrate, convolution_square_root
except ImportError:
    print('BoxFunction, boxconv, boxinner, boxnorm, boxintegrate, convolution_square_root, not loaded')

try:
    from .pointwise_observation_matrix import pointwise_observation_matrix, points_inside_mesh, PointwiseObservationOperator
except ImportError:
    print('pointwise_observation_matrix, points_inside_mesh, GridTransferOperator not loaded')