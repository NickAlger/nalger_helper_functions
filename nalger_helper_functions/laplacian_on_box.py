import numpy as np
import scipy.sparse as sps
import typing as typ
import functools as ft

__all__ = [
    'laplacian_1d',
    'dirichlet_laplacian_1d',
    'neumann_laplacian_1d',
    'periodic_laplacian_1d',
    'laplacian_nd',
    'dirichlet_laplacian_nd',
    'neumann_laplacian_nd',
    'periodic_laplacian_nd',
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
    _periodic_labels = {'P', 'p', 'Periodic', 'periodic', 1}
    if (((left_bc in _periodic_labels) and (right_bc not in _periodic_labels)) or
        ((left_bc not in _periodic_labels) and (right_bc in _periodic_labels))):
        raise RuntimeWarning('Asked to create Laplacian with one-sided periodicity. Probably not what you want.')

    num_cells = num_gridpoints - 1
    h = interval_width / num_cells
    L = dirichlet_laplacian_1d(num_gridpoints, interval_width).tolil()
    if left_bc in _neumann_labels:
        L[0,0] = 1.0 / h**2
    elif left_bc in _periodic_labels:
        L[0,-1] = -1.0 / h**2

    if right_bc in _neumann_labels:
        L[-1, -1] = 1.0 / h**2
    elif right_bc in _periodic_labels:
        L[-1,0] = -1.0 / h**2

    return L.tocsr()


neumann_laplacian_1d = ft.partial(laplacian_1d, left_bc='N', right_bc='N')
periodic_laplacian_1d = ft.partial(laplacian_1d, left_bc='P', right_bc='P')


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
        Li = sps.eye(1)
        for jj in range(d):
            if jj == ii:
                X = laplacian_1d(grid_shape[jj], box_widths[jj], left_bcs[jj], right_bcs[jj])
            else:
                X = sps.eye(grid_shape[jj]).tocsr()
            Li = sps.kron(Li, X, format='csr')
        L += Li.tocsr()
    return L


def dirichlet_laplacian_nd(
        grid_shape: typ.Sequence[int],      # [num_cells_x+1, num_cells_y+1, ...]
        box_widths: typ.Sequence[float],    # [width_x, width_y, ...]
) -> sps.csr_matrix:
    bcs = ['D' for _ in range(len(grid_shape))]
    return laplacian_nd(grid_shape, box_widths, bcs, bcs)


def neumann_laplacian_nd(
        grid_shape: typ.Sequence[int],      # [num_cells_x+1, num_cells_y+1, ...]
        box_widths: typ.Sequence[float],    # [width_x, width_y, ...]
) -> sps.csr_matrix:
    bcs = ['N' for _ in range(len(grid_shape))]
    return laplacian_nd(grid_shape, box_widths, bcs, bcs)


def periodic_laplacian_nd(
        grid_shape: typ.Sequence[int],      # [num_cells_x+1, num_cells_y+1, ...]
        box_widths: typ.Sequence[float],    # [width_x, width_y, ...]
) -> sps.csr_matrix:
    bcs = ['P' for _ in range(len(grid_shape))]
    return laplacian_nd(grid_shape, box_widths, bcs, bcs)

