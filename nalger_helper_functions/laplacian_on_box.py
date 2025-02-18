import numpy as np
import scipy.sparse as sps
import typing as typ

__all__ = [
    'laplacian_1d',
    'dirichlet_laplacian_1d',
    'neumann_laplacian_1d',
    'laplacian_nd',
    'dirichlet_laplacian_nd',
    'neumann_laplacian_nd',
]

def dirichlet_laplacian_1d(
        num_gridpoints: int, # num_cells + 1
        interval_width: float,
) -> sps.csr_matrix:
    '''-Matrix for -d^2/dx^2 with Dirichlet boundary conditions'''
    num_cells = num_gridpoints - 1
    h = interval_width / num_cells
    u = 2.0 * np.ones(num_gridpoints) / h**2
    v = -1.0 * np.ones(num_gridpoints-1) / h**2
    L = sps.diags([v, u, v], [-1, 0, 1]).tocsr()
    return L


def laplacian_1d(
        num_gridpoints: int,    # num_cells + 1
        interval_width: float,
        left_bc,           # 'D=Dirichlet=0 (default), N=Neumann
        right_bc,          # 'D=Dirichlet=1 (default), N=Neumann
) -> sps.csr_matrix:
    '''-Matrix for -d^2/dx^2
    Note: returns negative of PDE Laplacian. Returned matrix is positive, whereas PDE Laplacian is negative
    '''
    _neumann_labels = {'N', 'n', 'Neumann', 'neumann', 1}
    num_cells = num_gridpoints - 1
    h = interval_width / num_cells
    L = dirichlet_laplacian_1d(num_gridpoints, interval_width)
    if left_bc in _neumann_labels:
        L[0,0] = 1.0 / h**2

    if right_bc in _neumann_labels:
        L[-1, -1] = 1.0 / h**2

    return L


def neumann_laplacian_1d(
        num_gridpoints: int, # num_cells + 1
        interval_width: float,
) -> sps.csr_matrix:
    '''-Matrix for -d^2/dx^2 with Neumann boundary conditions'''
    return laplacian_1d(num_gridpoints, interval_width, 'Neumann', 'Neumann')


def laplacian_nd(
        grid_shape: typ.Sequence[int],      # [num_cells_x+1, num_cells_y+1, ...]
        box_widths: typ.Sequence[float],    # [width_x, width_y, ...]
        left_bcs:   typ.Sequence,           # [left_bc_x, left_bc_y, ...], 'D=Dirichlet=0 (default), N=Neumann=1
        right_bcs:  typ.Sequence,           # [right_bc_x, right_bc_y, ...]
) -> sps.csr_matrix:
    '''-Matrix for -(d^2/dx^2 + d^2/dy^2 + ...)
    Note: returns negative of PDE Laplacian. Returned matrix is positive, whereas PDE Laplacian is negative
    '''
    # see https://en.wikipedia.org/wiki/Kronecker_sum_of_discrete_Laplacians
    d = len(grid_shape)
    N = np.prod(grid_shape)
    L = sps.csr_matrix((N,N))
    for ii in range(d):
        Li = sps.eye(1).tocsr()
        for jj in range(d):
            if jj == ii:
                X = laplacian_1d(grid_shape[jj], box_widths[jj], left_bcs[jj], right_bcs[jj])
            else:
                X = sps.eye(grid_shape[jj]).tocsr()
            Li = sps.kron(Li, X, format='csr')
        L += Li
    return L


def dirichlet_laplacian_nd(
        grid_shape: typ.Sequence[int],      # [num_cells_x+1, num_cells_y+1, ...]
        box_widths: typ.Sequence[float],    # [width_x, width_y, ...]
) -> sps.csr_matrix:
    '''-Matrix for -(d^2/dx^2 + d^2/dy^2 + ...) with Dirichlet B.C.'s.
    Note: returns negative of PDE Laplacian. Returned matrix is positive, whereas PDE Laplacian is negative
    '''
    d = len(grid_shape)
    bcs = [0 for _ in range(d)]
    return laplacian_nd(grid_shape, box_widths, bcs, bcs)


def neumann_laplacian_nd(
        grid_shape: typ.Sequence[int],      # [num_cells_x+1, num_cells_y+1, ...]
        box_widths: typ.Sequence[float],    # [width_x, width_y, ...]
) -> sps.csr_matrix:
    '''-Matrix for -(d^2/dx^2 + d^2/dy^2 + ...) with Neumann B.C.'s.
    Note: returns negative of PDE Laplacian. Returned matrix is positive, whereas PDE Laplacian is negative
    '''
    d = len(grid_shape)
    bcs = [1 for _ in range(d)]
    return laplacian_nd(grid_shape, box_widths, bcs, bcs)

