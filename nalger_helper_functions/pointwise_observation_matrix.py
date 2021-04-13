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
        # Loop over all interpolation points
        x = np.copy(pp[k, :])
        p = dl.Point(x[0], x[1])
        # Find cell for the point
        cell_id = bbt.compute_first_entity_collision(p)
        # Vertex coordinates for the cell
        xvert = coords[cells[cell_id, :], :]
        # Evaluate the basis functions for the cell at x
        v = dolfin_element.evaluate_basis_all(x, xvert, cell_id)
        jj = np.arange(sdim * k, sdim * (k + 1))
        rows[jj] = k
        # Find the dofs for the cell
        cols[jj] = dofmap.cell_dofs(cell_id)
        vals[jj] = v

    ij = np.concatenate((np.array([rows]), np.array([cols])), axis=0)
    M = sps.csr_matrix((vals, ij), shape=(nx, V.dim()))
    return M


mesh = dl.UnitSquareMesh(11,13)
V = dl.FunctionSpace(mesh, 'CG', 2)
u = dl.Function(V)
u.vector()[:] = np.random.randn(V.dim())

N = 100
d = mesh.geometric_dimension()
pp = np.random.rand(N, d)
B = pointwise_observation_matrix(pp, V)

x1 = np.zeros(N)
for k in range(N):
    x1[k] = u(dl.Point(pp[k,:]))

x2 = B * u.vector()[:]

err_pointwise_observation_matrix = np.linalg.norm(x2 - x1)
print('err_pointwise_observation_matrix=', err_pointwise_observation_matrix)
