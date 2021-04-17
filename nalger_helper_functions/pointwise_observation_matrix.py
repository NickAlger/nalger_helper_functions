import numpy as np
import scipy.sparse as sps
import dolfin as dl


def pointwise_observation_matrix(pp, V, nonzero_columns_only=False, return_inside_mesh_mask=False):
    inside_mesh_mask = points_inside_mesh(pp, V.mesh())
    inside_inds = np.argwhere(inside_mesh_mask).reshape(-1)
    qq = pp[inside_inds, :]

    B0 = pointwise_observation_matrix_interior_points_only(qq, V)
    if nonzero_columns_only:
        nonzero_cols = dofs_that_contribute_to_function_at_points(qq, V)
        B0 = B0[:, nonzero_cols]
        # B0 = B0.tocsr()

    B0_diffs = B0.indptr[1:] - B0.indptr[0:-1]
    B_diffs = np.zeros(pp.shape[0]+1, dtype=B0.indptr.dtype)
    B_diffs[inside_inds+1] = B0_diffs
    B_indptr = np.cumsum(B_diffs)

    B = sps.csr_matrix((B0.data, B0.indices, B_indptr), (pp.shape[0], B0.shape[1]))

    if nonzero_columns_only and return_inside_mesh_mask:
        return B, nonzero_cols, inside_mesh_mask
    elif nonzero_columns_only:
        return B, nonzero_cols
    elif return_inside_mesh_mask:
        return B, inside_mesh_mask
    else:
        return B


def pointwise_observation_matrix_interior_points_only(pp, V):
    """
    pp = Nxd array of N points in d dimensions
    V = FunctionSpace
    Based on code by Stefan_Jakobsson, posted online at:
    http://fenicsproject.org/qa/4372/interpolator-as-matrix?show=4418#a4418
    """
    nx, dim = pp.shape
    mesh = V.mesh()
    coords = mesh.coordinates()
    cells = mesh.cells()
    dolfin_element = V.dolfin_element()
    dofmap = V.dofmap()
    bbt = mesh.bounding_box_tree()
    sdim = dolfin_element.space_dimension()
    rows = np.zeros(nx * sdim, dtype='int')
    cols = np.zeros(nx * sdim, dtype='int')
    vals = np.zeros(nx * sdim)
    for k in range(nx):
        # Loop over all observation points
        x = np.copy(pp[k, :])
        p = dl.Point(x)
        jj = np.arange(sdim * k, sdim * (k + 1))
        # Find cell for the point
        cell_id = bbt.compute_first_entity_collision(p)
        # Vertex coordinates for the cell
        xvert = coords[cells[cell_id, :], :]
        # Evaluate the basis functions for the cell at x
        vals[jj] = dolfin_element.evaluate_basis_all(x, xvert, cell_id)
        rows[jj] = k
        # Find the dofs for the cell
        cols[jj] = dofmap.cell_dofs(cell_id)

    ij = np.concatenate((np.array([rows]), np.array([cols])), axis=0)
    B = sps.csr_matrix((vals, ij), shape=(nx, V.dim()))
    return B


def points_inside_mesh(pp, mesh):
    bbt = mesh.bounding_box_tree()
    infinity_point = dl.Point(np.inf * np.ones(mesh.geometric_dimension()))
    outside_mesh_entity = bbt.compute_first_entity_collision(infinity_point)
    N, d = pp.shape
    outside_pts = np.zeros(N, dtype=bool)
    for k in range(N):
        p = np.copy(pp[k,:])
        cell_id = bbt.compute_first_entity_collision(dl.Point(p))
        outside_pts[k] = (cell_id == outside_mesh_entity)
    inside_pts = np.logical_not(outside_pts)
    return inside_pts


def dofs_that_contribute_to_function_at_points(pp, V):
    # degrees of freedom that contribute to function values at points pp
    mesh = V.mesh()
    dofmap = V.dofmap()
    qq = pp[points_inside_mesh(pp, mesh),:]
    N, d = qq.shape
    bbt = mesh.bounding_box_tree()
    contributing_dofs = list()
    for k in range(N):
        cell_id = bbt.compute_first_entity_collision(dl.Point(np.copy(qq[k, :])))
        contributing_dofs.append(dofmap.cell_dofs(cell_id))
    contributing_dofs = np.unique(np.concatenate(contributing_dofs)).reshape(-1)
    return contributing_dofs


