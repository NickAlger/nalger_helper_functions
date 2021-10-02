import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt

nx=13
ny=9
mesh = dl.UnitSquareMesh(nx,ny)

plt.figure()
dl.plot(mesh)

p = np.array([1.1, 0.05])
plt.plot(p[0], p[1], '*r')


tdim = mesh.topology().dim()
bbt = mesh.bounding_box_tree()
entity, distance = bbt.compute_closest_entity(dl.Point(p))

mesh.geometry()

C = mesh.cells()[entity]
V = mesh.coordinates()[C,:].T
plt.plot(V[0,:], V[1,:], '*b')

print(V)

# def projected_affine_coordinates_2(p, V):
#     v0 = V[:, 0].reshape(-1)
#     dV = V[:, 1].reshape(-1) - v0
#     dp = p.reshape(-1) - v0
#     c1 = np.dot(dV, dp) / np.dot(dV, dV)
#     c0 = 1. - c1
#     affine_coordinates = np.array([c0, c1])
#     return affine_coordinates

# def projected_affine_coordinates_2_vectorized(PP, VV):
#     num_pts, N = PP.shape
#     num_pts, k, N = VV.shape
#     if k != 2:
#         raise RuntimeError('this function is for 1D affine subspaces defined by 2 points')
#
#     VV0 = VV[:, 0, :].reshape((num_pts, 1, N))
#     dVV = VV[:, 1, :].reshape((num_pts, 1, N)) - VV0
#     dPP = PP.reshape((num_pts, 1, N)) - VV0
#     cc1 = np.sum(dVV * dPP, axis=2) / np.sum(dVV * dVV, axis=2).reshape((num_pts, 1))
#     cc0 = 1. - (1. - np.sum(cc1, axis=1)).reshape((num_pts, 1))
#     affine_coordinates = np.concatenate([cc0, cc1], axis=1)
#     return affine_coordinates

def projected_affine_coordinates_k(p, V):
    N, k = V.shape
    v0 = V[:, 0].reshape((N, 1))
    dV = V[:, 1:].reshape((N, -1)) - v0
    p_hat = p.reshape((N, -1)) - v0
    cc_rest = np.linalg.lstsq(dV, p_hat, rcond=None)[0].reshape(-1)
    c_first = np.array([1. - np.sum(cc_rest)])
    affine_coordinates = np.concatenate([c_first, cc_rest])
    return affine_coordinates

def projected_affine_coordinates_vectorized(PP, VV):
    num_pts, N = PP.shape
    num_pts, k, N = VV.shape

    if k == 1:
        affine_coordinates = np.ones((num_pts, 1))
    else:
        VV0 = VV[:, 0, :].reshape((num_pts, 1, N))
        dVV = VV[:, 1: :].reshape((num_pts, k-1, N)) - VV0
        dPP = PP.reshape((num_pts, 1, N)) - VV0

        if k == 2:
            PHI = np.sum(dVV * dVV, axis=-1).reshape((num_pts))
            RHS = np.sum(dVV * dPP, axis=-1).reshape((num_pts))
            cc_rest = (RHS / PHI).reshape((num_pts, 1))
        else:
            PHI = np.einsum('xiz,xjz->xij', dVV, dVV) # shape = (num_pts, k-1, k-1)
            iPHI = np.linalg.inv(PHI) # shape = (num_pts, k-1, k-1)

            RHS = np.sum(dVV * dPP, axis=-1)  # shape = (num_pts, k-1)
            cc_rest = np.einsum('pij,pj->pi', iPHI, RHS)  # shape = (num_pts, k-1)

        cc_first = (1. - np.sum(cc_rest, axis=1)).reshape((num_pts, 1))
        affine_coordinates = np.concatenate([cc_first, cc_rest], axis=1)

    return affine_coordinates # shape = (num_pts, k)


# def project_point_onto_affine_subspace(p, V):
#     # Projects point p onto the k-dimensional affine subspace
#     # that contains the columns of V
#     if len(V.shape) == 1:
#         V = V.reshape((-1,1))
#     N, k = V.shape
#
#     if k == 1:
#         affine_coordinates = np.array([1.])
#     elif k == 2:
#         affine_coordinates = projected_affine_coordinates_k(p, V)
#     else:
#         affine_coordinates = projected_affine_coordinates_k(p, V)
#     projected_p = np.dot(V, affine_coordinates).reshape(p.shape)
#     return projected_p, affine_coordinates


def project_point_onto_affine_subspace(p, V):
    # Projects a point onto an affine subspace
    #
    # p.shape = (N,)   OR (num_pts, N) for vectorization over many points / affine subspaces
    # V.shape = (k, N) OR (num_pts, k, N) for vectorization
    #
    # N = dimension of ambient space
    # k-1 = dimension of affine subspace
    #
    # For a single point / affine subspace:
    #   - p is the point in R^N to be projected onto the affine subspace
    #   - The affine subspace is the set of all linear combinations
    #   of rows, V[i,:], of the matrix V
    #
    # Vectorization:
    # For many points / affine subspaces, each point is
    # projected onto its corresponding affine subspace
    #   p[i,:] is the ith point
    #   V[i,:,:] is the matrix defining the ith affine subspace
    if len(p.shape) == 1:
        PP = p.reshape((1, p.shape[0]))
        VV = V.reshape((1, V.shape[0], V.shape[1]))
    else:
        PP = p
        VV = V
    num_pts, k, N = VV.shape

    if k == 1:
        affine_coordinates = np.ones((num_pts, 1))
    else:
        VV0 = VV[:, 0, :].reshape((num_pts, 1, N))
        dVV = VV[:, 1: :].reshape((num_pts, k-1, N)) - VV0
        dPP = PP.reshape((num_pts, 1, N)) - VV0

        if k == 2:
            PHI = np.sum(dVV * dVV, axis=-1).reshape((num_pts))
            RHS = np.sum(dVV * dPP, axis=-1).reshape((num_pts))
            cc_rest = (RHS / PHI).reshape((num_pts, 1))
        else:
            PHI = np.einsum('xiz,xjz->xij', dVV, dVV) # shape = (num_pts, k-1, k-1)
            iPHI = np.linalg.inv(PHI) # shape = (num_pts, k-1, k-1)

            RHS = np.sum(dVV * dPP, axis=-1)  # shape = (num_pts, k-1)
            cc_rest = np.einsum('pij,pj->pi', iPHI, RHS)  # shape = (num_pts, k-1)

        cc_first = (1. - np.sum(cc_rest, axis=1)).reshape((num_pts, 1))
        affine_coordinates = np.concatenate([cc_first, cc_rest], axis=1) # shape = (num_pts, k)

    PP_projected = np.einsum('pi,pij->pj', affine_coordinates, VV) # shape = (num_pts, N)
    return PP_projected, affine_coordinates


# def closest_point_on_triangle(p, V):
#     # Finds the point q on the triangle generated by
#     # convex combinations of the columns of V
#     # that is closest to the point p.
#     # V.shape = (N, 3)
#     # p.shape = (N,)
#     q0, c0 = project_point_onto_affine_subspace(p, V[:,0])
#     q1, c1 = project_point_onto_affine_subspace(p, V[:,1])
#     q2, c2 = project_point_onto_affine_subspace(p, V[:,2])
#
#     q01, c01 = project_point_onto_affine_subspace(p, V[:,[0,1]])
#     q02, c02 = project_point_onto_affine_subspace(p, V[:,[0,2]])
#     q12, c12 = project_point_onto_affine_subspace(p, V[:,[1,2]])
#
#     q123, c123 = project_point_onto_affine_subspace(p, V)
#
#     qq = [q0, q1, q2, q01, q02, q12, q123]
#     cc = [c0, c1, c2, c01, c02, c12, c123]
#
#     in_triangle_dd = list()
#     in_triangle_qq = list()
#     in_triangle_cc = list()
#     for c, q in zip(cc, qq):
#         if np.all(0. <= c) and np.all(c <= 1.):
#             in_triangle_qq.append(q)
#             in_triangle_cc.append(c)
#             d = np.linalg.norm(q - p)
#             in_triangle_dd.append(d)
#
#     closest_ind = np.argmin(in_triangle_dd)
#     closest_point = in_triangle_qq[closest_ind]
#     closest_convex_coordinates = in_triangle_cc[closest_ind]
#     closest_distance = in_triangle_dd[closest_ind]
#     return closest_point, closest_convex_coordinates, closest_distance


def powerset(s):
    # https://stackoverflow.com/a/1482320/484944
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


def closest_point_on_simplex(p, V):
    """Projects a point onto a simplex (triangle, tetrahedron, etc)

    p.shape = (N,)   OR (num_pts, N) for vectorization over many points/simplices
    V.shape = (k, N) OR (num_pts, k, N) for vectorization

    N = dimension of ambient space
    k-1 = dimension of simplex

    For a single point/simplex:
        - p is the point in R^N to be projected onto the simplex
        - The simplex is the set of all convex combinations
        of rows, V[i,:], of the matrix V

    Vectorization:
    For many points/simplices, each point is
    projected onto its corresponding simplex
        p[i,:] is the ith point
        V[i,:,:] is the matrix defining the ith simplex

    Example usage:
        import numpy as np
        import matplotlib.pyplot as plt
        p1 = np.array([1.1, 0.4])
        V1 = np.array([[0., 0.],
                       [0., 1.],
                       [1., 0.]])
        p2 = np.array([-0.3, 1.1])
        V2 = np.array([[-1.0, 0.],
                       [0.,   0.],
                       [-0.5, 0.5]])
        p = np.stack([p1, p2])
        V = np.stack([V1, V2])
        projected_p = closest_point_on_simplex(p, V)
        projected_p1 = projected_p[0,:]
        projected_p2 = projected_p[1,:]
        plt.figure()
        t1 = plt.Polygon(V1)
        plt.gca().add_patch(t1)
        plt.plot([p1[0], projected_p1[0]],
                 [p1[1], projected_p1[1]], 'r')
        t2 = plt.Polygon(V2)
        plt.gca().add_patch(t2)
        plt.plot([p2[0], projected_p2[0]],
                 [p2[1], projected_p2[1]], 'r')
        plt.gca().set_aspect('equal')
        plt.show()
    """
    if len(p.shape) == 1:
        PP = p.reshape((1, p.shape[0]))
        VV = V.reshape((1, V.shape[0], V.shape[1]))
    else:
        PP = p
        VV = V
    num_pts, k, N = VV.shape

    subsets = list(powerset(list(range(k)))) # e.g., [[], [0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]]
    QQ = list()
    CC = list()
    for s in subsets:
        if s:
            simplicial_facet = VV[:,s,:]
            Q, C = project_point_onto_affine_subspace(PP, simplicial_facet)
            QQ.append(Q)
            CC.append(C)

    distances = np.stack([np.linalg.norm(Q - PP, axis=-1) for Q in QQ]) # shape=(num_subsets, num_pts)

    good_inds = np.stack([(np.all(0. <= C, axis=1) & np.all(C <= 1., axis=1)) for C in CC]) # shape=(num_subsets, num_pts)
    bad_inds = np.logical_not(good_inds)
    distances[bad_inds] = np.inf
    closest_inds = np.expand_dims(np.argmin(distances[:, :, None], axis=0), axis=0)

    QQ_stack = np.stack(QQ) # shape=(7, num_pts, N)
    PP_projected = np.take_along_axis(QQ_stack, closest_inds, axis=0)[0,:,:]

    if len(p.shape) == 1:
        PP_projected = PP_projected.reshape(-1)
    return PP_projected



q_min, c_min, d_min = closest_point_on_triangle(p, V)

print('q_min=', q_min)

q_min2 = closest_point_on_simplex(p, V.T)

print('q_min2=', q_min2)

from time import time


num_pts = 10000
pp = np.random.randn(num_pts, 2)
t = time()
for k in range(num_pts):
    pk = pp[k,:]
    entity, distance = bbt.compute_closest_entity(dl.Point(pk))
    C = mesh.cells()[entity]
    Vk = mesh.coordinates()[C, :].T
    q_mink, c_mink, d_mink = closest_point_on_triangle(pk, Vk)
dt = time() - t
print('dt=', dt)



d=len(p)
# Check edge 0-1
p_hat = p - V[:,0]
V_hat = V[:,1].reshape((d,-1)) - V[:,0].reshape((d,-1))
c_01 = np.linalg.lstsq(V_hat, p_hat, rcond=None)[0]
print('c_01=', c_01)

projected_p01, c01 = project_point_onto_affine_subspace(p, V[:,[0,1]])

p_hat = p - V[:,0]
V_hat = V[:,2].reshape((2,-1)) - V[:,0].reshape((2,-1))
c_02 = np.linalg.lstsq(V_hat, p_hat, rcond=None)[0]
print('c_02=', c_02)

projected_p02, c02 = project_point_onto_affine_subspace(p, V[:,[0,2]])

p_hat = p - V[:,1]
V_hat = V[:,2].reshape((2,-1)) - V[:,1].reshape((2,-1))
c_12 = np.linalg.lstsq(V_hat, p_hat, rcond=None)[0]
print('c_12=', c_12)

projected_p12, c12 = project_point_onto_affine_subspace(p, V[:,[1,2]])

k=2
num_pts = 50000
N=3
PP = np.random.randn(num_pts, N)
VV = np.random.randn(num_pts, k, N)
t = time()
cc = projected_affine_coordinates_vectorized(PP, VV)
dt_vectorized = time() - t
print('dt_vectorized=', dt_vectorized)

t = time()
cc_true = list()
for ii in range(num_pts):
    p = PP[ii,:]
    V = VV[ii, :, :].T
    _, c_true = project_point_onto_affine_subspace(p, V)
    cc_true.append(c_true)
cc_true = np.array(cc_true)
dt_unvectorized = time() - t
print('dt_unvectorized=', dt_unvectorized)

err_vectorized = np.linalg.norm(cc_true - cc)
print('err_vectorized=', err_vectorized)

k=3
num_pts = 50000
N=3
PP = np.random.randn(num_pts, N)
VV = np.random.randn(num_pts, k, N)
t = time()
cc = projected_affine_coordinates_vectorized(PP, VV)
dt_vectorized = time() - t
print('dt_vectorized=', dt_vectorized)

t = time()
cc_true = list()
for ii in range(num_pts):
    _, c_true = project_point_onto_affine_subspace(PP[ii,:], VV[ii, :, :].T)
    cc_true.append(c_true)
cc_true = np.array(cc_true)
dt_unvectorized = time() - t
print('dt_unvectorized=', dt_unvectorized)

err_vectorized = np.linalg.norm(cc_true - cc) / np.linalg.norm(cc_true)
print('err_vectorized=', err_vectorized)

# k, num_pts, N = VV.shape
# VV0 = VV[0, :, :].reshape((1, num_pts, N))
# dVV = VV[1:, :, :].reshape((k - 1, num_pts, N)) - VV0
# dPP = PP.reshape((1, num_pts, N)) - VV0
# RHS = np.sum(dVV * dPP, axis=-1)  # shape = (k-1, num_pts)
# PHI = np.einsum('ixz,jxz->ijx', dVV, dVV).swapaxes(0, 2)  # shape = (num_pts, k-1, k-1)
# iPHI = np.linalg.inv(PHI)  # shape = (num_pts, k-1, k-1)
# cc_rest = np.einsum('pij,jp->pi', iPHI, RHS)  # shape = (num_pts, k-1)
# cc_first = (1. - np.sum(cc_rest, axis=1)).reshape((num_pts, 1))
# affine_coordinates = np.concatenate([cc_first, cc_rest], axis=1)


