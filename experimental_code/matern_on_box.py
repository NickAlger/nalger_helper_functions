import numpy as np
from scipy.special import gamma as gamma_func
import typing as typ

from nalger_helper_functions.laplacian_on_box import laplacian_nd
from nalger_helper_functions.mass_matrix_on_box import mass_matrix_nd


def matern_covariance_inverse_square_root(
        grid_shape: typ.Sequence[int],
        box_widths: typ.Sequence[float],
        characteristic_length: float,
        pointwise_standard_deviation: float,
        left_bcs: typ.Sequence, # 0='D'='Dirichlet, 1='N'='Neumann'
        right_bcs: typ.Sequence,
):
    '''Inverse square root of Matern covariance operator.
        A = -gamma*Laplacian + alpha = gamma*K + alpha * M
    Case p=2 in:
        Daon, Yair, and Georg Stadler. "Mitigating the influence of the boundary on PDE-based covariance operators.".

    Note: here we do not have the Robin B.C. method implemented

    characteristic_length = C1 sqrt(gamma/alpha) where C1 = sqrt(8 nu) where nu = p - d/2
                          = distance at which covariance decays by 90% from its peak
                          (page 5 in Daon and Stadler, in paragraph near the top of the page)
                          (also in text on page 426 in Lindgren, Rue, Lindstrom)

    pointwise_variance = C2 /(alpha^nu gamma^(d/2)) where C2 = Gamma(nu)/(Gamma(nu+d/2) (4pi)^(d/2))
                       = pointwise_standard_deviation^2
                       = diagonal of A^-1 M A^-1
                         (equation 15 in Appendix B in Daon and Stadler, page 19)

    '''
    M = mass_matrix_nd(grid_shape, box_widths)
    K = laplacian_nd(grid_shape, box_widths, left_bcs, right_bcs)

    p = 2.0
    d = len(grid_shape)
    nu = p - d/2.0
    C1 = np.sqrt(8.0 * nu)
    C2 = gamma_func(nu) / (gamma_func(nu+d/2.0) * np.power(4.0*np.pi,d/2))
    C3 = characteristic_length / C1
    pointwise_variance = pointwise_standard_deviation**2

    alpha = np.power(C2 / (pointwise_variance * C3**d), 1.0 / p)
    gamma = alpha * C3**2

    gamma = 1.0
    alpha = 8 * nu / characteristic_length**2

    A = gamma * K + alpha * M
    return A, (gamma, alpha)


import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

grid_shape = (193, 188)
box_widths = (4.0, 3.0)

characteristic_length = 0.04
pointwise_standard_deviation = 1.0

A, (gamma, alpha) = matern_covariance_inverse_square_root(
    grid_shape, box_widths, characteristic_length, pointwise_standard_deviation,
    ['N', 'N'], ['N', 'N'],
    # ['D', 'D'], ['D', 'D'],
)

M = mass_matrix_nd(grid_shape, box_widths)


# M = mass_matrix_nd(grid_shape, box_widths)
# # K = laplacian_nd(grid_shape, box_widths, ['N', 'N'], ['N', 'N'])
# K = laplacian_nd(grid_shape, box_widths, ['D', 'D'], ['D', 'D'])

# p = 2.0
# d = len(grid_shape)
# nu = p - d / 2.0
# C1 = np.sqrt(8.0 * nu)
#
# alpha = 3.0
# gamma = alpha * (0.5 / C1)**2
#
# gamma = characteristic_length**2
# A = 0.01 * K + 1.0 * M
# A = alpha * K + gamma * M



n = np.prod(grid_shape)


mid_ii = int(0.5 * grid_shape[0])
mid_jj = int(0.5 * grid_shape[1])
mid_ind = grid_shape[1]*mid_ii + mid_jj
e = np.zeros(n)
e[mid_ind] = 1.0

# phi = spla.spsolve(A, M @ spla.spsolve(A, e)).reshape(grid_shape)
phi = (M @ spla.spsolve(A, e)).reshape(grid_shape)

if False:
    plt.figure()
    plt.imshow(phi.T, origin='lower', extent=[0, box_widths[0], 0, box_widths[1]])

xx = np.linspace(0, box_widths[0], grid_shape[0])
yy = np.linspace(0, box_widths[1], grid_shape[1])

phi_slice = phi[mid_ii,:]

normalized_phi_slice = phi_slice - np.min(phi_slice)
normalized_phi_slice = normalized_phi_slice / np.max(normalized_phi_slice)

plt.plot(yy, normalized_phi_slice)
# plt.plot(yy, phi_slice)

plt.plot(yy, 0.1*np.ones(len(yy)))
