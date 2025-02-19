import numpy as np
import scipy.sparse as sps
import typing as typ
import functools as ft


__all__ = [
    'mass_matrix_1d',
    'mass_matrix_nd',
]


def mass_matrix_1d(
        num_gridpoints: int, # num_cells + 1
        interval_width: float,
        trapezoid_rule: bool = False,
) -> sps.csr_matrix:
    '''Creates mass matrix M such that sum(M @ u_vec) =approx= int_a^b u(x) dx for function u on interval [a,b].
    interval_width = b-a
    '''
    num_cells = num_gridpoints - 1
    h = interval_width / num_cells

    mm = h * np.ones(num_gridpoints)
    if trapezoid_rule:
        mm[0] = h/2
        mm[-1] = h/2

    M = sps.diags([mm], [0])
    return M


def mass_matrix_nd(
        grid_shape: typ.Sequence[int],
        box_widths: typ.Sequence[float],
        trapezoid_rule: bool = False,
) -> sps.csr_matrix:
    '''Creates mass matrix M such that sum(M @ u_vec) =approx= int_ax^bx int_ay^by ... u(x,y,...) dx
    for function u on box [ax,bx] x [ay, by] x ... .
    '''
    d = len(grid_shape)
    M = sps.eye(1).tocsr()
    for ii in range(d):
        Mi = mass_matrix_1d(grid_shape[ii], box_widths[ii], trapezoid_rule)
        M = sps.kron(M, Mi)
    return M

