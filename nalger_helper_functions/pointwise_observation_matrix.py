import numpy as np
import scipy.sparse as sps
import dolfin as dl


def pointwise_observation_matrix(pp, V):
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


class PointwiseObservationOperator:
    def __init__(me, pp, V, exterior_fill_value=0.0, inside_mesh_mask=None, inside_mesh_transfer_matrix=None,
                 use_transpose=False):
        me.pp = pp
        me.V = V
        me.exterior_fill_value = exterior_fill_value
        me.use_transpose = use_transpose

        num_pts, me.ndim = me.pp.shape
        if use_transpose:
            me.shape = (V.dim(), num_pts)
        else:
            me.shape = (num_pts, V.dim())

        if inside_mesh_mask is None:
            inside_mesh_mask = points_inside_mesh(pp, V.mesh())
        me.inside_mesh_mask = inside_mesh_mask

        if inside_mesh_transfer_matrix is None:
            inside_mesh_transfer_matrix = pointwise_observation_matrix(pp[me.inside_mesh_mask, :], V)
        me.inside_mesh_transfer_matrix = inside_mesh_transfer_matrix

    def _forward_matvec(me, u):
        v = me.exterior_fill_value * np.ones(me.shape[0], dtype=u.dtype)
        v[me.inside_mesh_mask] = me.inside_mesh_transfer_matrix * u
        return v

    def _transpose_matvec(me, v):
        return me.inside_mesh_transfer_matrix.T * v[me.inside_mesh_mask]

    def matvec(me, x):
        if me.use_transpose:
            return me._transpose_matvec(x)
        else:
            return me._forward_matvec(x)

    def rmatvec(me, x):
        if me.use_transpose:
            return me._forward_matvec(x)
        else:
            return me._transpose_matvec(x)

    def __mul__(me, x):
        return me.matvec(x)

    def __rmul__(me, x):
        return me.rmatvec(x)

    def __matmul__(me, x):
        return me.matvec(x)

    def __rmatmul__(me, x):
        return me.rmatvec(x)

    def transpose(me):
        return PointwiseObservationOperator(me.pp, me.V, exterior_fill_value=me.exterior_fill_value,
                                            inside_mesh_mask=me.inside_mesh_mask,
                                            inside_mesh_transfer_matrix=me.inside_mesh_transfer_matrix,
                                            use_transpose= not me.use_transpose)

    @property
    def T(me):
        return me.transpose()
