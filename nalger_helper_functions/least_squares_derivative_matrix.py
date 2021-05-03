import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree


_shared_vars = dict()

def complex_exponential(vhat, omega, points):
    s = 2. * np.pi * omega * 1j
    return np.exp(s * np.dot(points, vhat))


def directional_derivative_of_complex_exponential(vhat, omega, points, derivative_direction):
    s = 2. * np.pi * omega * 1j
    return s * np.dot(derivative_direction, vhat) * np.exp(s * np.dot(points, vhat))


def least_squares_directional_derivative_matrix(points, derivative_direction,
                                                a_reg=1e-8, num_neighbors=10,
                                                num_angles=5, num_frequencies=4,
                                                s_min=2, s_max=20):
    N, d = points.shape

    uhat = derivative_direction / np.linalg.norm(derivative_direction)

    T = cKDTree(points)
    neighbor_distances, neighbor_inds = T.query(points, num_neighbors)

    rows = list() # [0,    0,    0,     1,    1,    1,     2,    2,    2,     ... ]
    cols = list() # [p0_n0,p0_n1,p0_n2, p1_n0,p1_n1,p1_n2, p2_n0,p2_n1,p2_n2, ... ]
    values = list()
    for r in range(N): # for each row
        cc = neighbor_inds[r, :] # numpy array of ints. shape = (num_nbrs,)
        rows += [r for _ in range(len(cc))]
        cols += list(cc)

        pr = points[r, :].reshape((1, -1))
        pp_nbrs = points[cc, :] # numpy array of floats. shape = (num_nbrs, spatial_dimension)
        dd_nbrs = neighbor_distances[r, :] # numpy array of floats. shape = (num_nbrs,)
        local_pointcloud_diameter = 2.0 * np.max(dd_nbrs)
        min_L = s_min * local_pointcloud_diameter
        max_L = s_max * local_pointcloud_diameter

        theta0 = np.arctan(uhat[1] / uhat[0])
        thetas = theta0 + np.linspace(0, 2.*np.pi, num_angles, endpoint=False)
        vhats = np.zeros((num_angles, d))
        vhats[:, 0] = np.cos(thetas)
        vhats[:, 1] = np.sin(thetas)

        omegas = 1. / np.linspace(min_L, max_L, num_frequencies)

        X = np.zeros((num_neighbors, num_angles, num_frequencies, 2))
        DXr = np.zeros((num_angles, num_frequencies,2))
        for ii in range(num_angles):
            for jj in range(num_frequencies):
                vhat = vhats[ii,:]
                omega = omegas[jj]
                X_ij = complex_exponential(vhat, omega, pp_nbrs) / omega
                X[:, ii, jj, 0] = X_ij.real
                X[:, ii, jj, 1] = X_ij.imag

                DXr_ij = directional_derivative_of_complex_exponential(vhat, omega, pr, derivative_direction) / omega
                DXr[ii, jj, 0] = DXr_ij.real
                DXr[ii, jj, 1] = DXr_ij.imag

        X = X.reshape((num_neighbors, -1))
        DXr = DXr.reshape(-1)

        A = np.bmat([[X.T], [a_reg*np.eye(len(cc))]])
        b = np.concatenate([DXr, np.zeros(len(cc))])

        q = np.linalg.lstsq(A, b, rcond=None)[0]  # min 0.5*||q^T*X - DXr||^2 + a_reg*0.5*||q||^2
        values += list(q)

    rows = np.array(rows)
    cols = np.array(cols)
    values = np.array(values)

    D = sps.coo_matrix((values, (rows, cols)), shape=(N, N)).tocsr()

    return D


def least_squares_directional_derivative_matrix_old(points, derivative_direction,
                                                a_reg=1e-6, num_neighbors=10,
                                                num_angles=5, num_frequencies=4,
                                                min_points_per_wavelength=30,
                                                plot_probing_functions=False,
                                                apply_D_true=None,
                                                run_finite_difference_checks=False):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/least_squares_derivative_matrix.ipynb
    N, d = points.shape
    nnz = N * num_neighbors


    T = cKDTree(points)
    neighbor_distances, neighbor_inds = T.query(points, num_neighbors)

    rows = np.outer(np.arange(N, dtype=int), np.ones(num_neighbors, dtype=int)).reshape(-1)  # length nnz
    cols = neighbor_inds.reshape(-1)  # length nnz

    min_h = np.min(neighbor_distances[:, 1])
    mesh_diameter = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
    print('min_h=', min_h, ', mesh_diameter=', mesh_diameter)

    max_L = mesh_diameter
    # min_L = min_points_per_wavelength * min_h
    min_L = max_L / 2.

    uhat = derivative_direction / np.linalg.norm(derivative_direction)

    theta0 = np.arctan(uhat[0] / uhat[1])
    thetas = theta0 + np.linspace(0, 2.*np.pi, num_angles, endpoint=False)
    vhats = np.zeros((num_angles, d))
    vhats[:, 0] = np.cos(thetas)
    vhats[:, 1] = np.sin(thetas)

    # vhats = np.zeros((num_angles, d))
    # vhats[0, :] = uhat
    # for k in range(1, num_angles):
    #     while True:
    #         v = np.random.randn(d)
    #         vhat = v / np.linalg.norm(v)
    #         if np.dot(vhat, uhat) > orthtol:
    #             break
    #     vhats[k, :] = vhat

    omegas = 1. / np.linspace(min_L, max_L, num_frequencies)

    X = np.zeros((N, num_angles, num_frequencies), dtype=complex)
    Y = np.zeros((N, num_angles, num_frequencies), dtype=complex)
    for ii in range(num_angles):
        for jj in range(num_frequencies):
            X[:, ii, jj] = complex_exponential(vhats[ii, :], omegas[jj], points)
            Y[:, ii, jj] = directional_derivative_of_complex_exponential(vhats[ii, :], omegas[jj], points,
                                                                          derivative_direction)

    X = X.reshape((N, -1))
    Y = Y.reshape((N, -1))

    _shared_vars['a_reg'] = a_reg
    _shared_vars['N'] = N
    _shared_vars['nnz'] = nnz
    _shared_vars['rows'] = rows
    _shared_vars['cols'] = cols
    _shared_vars['X'] = X
    _shared_vars['Y'] = Y

    print('nnz=', nnz, ', np.size(X)=', np.size(X))

    if plot_probing_functions:
        if apply_D_true is not None:
            Y_true = np.zeros(Y.shape)
            for k in range(Y.shape[1]):
                Y_true[:,k] = apply_D_true(X[:,k])
        else:
            Y_true = None

        make_probing_function_plots(points, X, Y, Y_true)

    # Hessian
    HH = list()
    for k in range(X.shape[1]):
        Jk = D_prime_mode3_contraction(X[:, k])
        Hk = Jk.H * Jk
        HH.append(Hk.real)

    Hd = HH[0]
    for Hdk in HH[1:]:
        Hd += Hdk

    Hr = a_reg * sps.eye(Hd.shape[0]).tocsr()
    H = Hd + Hr  # Hessian doesn't change

    _shared_vars['H'] = H

    if run_finite_difference_checks:
        finite_difference_check_D_prime()
        finite_difference_check_gradient_and_hessian()

    # Solve for optimal matrix values q
    g0 = gradient(np.zeros(nnz))
    q = spla.spsolve(H, -g0)

    return make_D_matrix(q).real


def make_probing_function_plots(points, X, Y, Y_true):
    for k in range(X.shape[1]):
        if Y_true is not None:
            fig = plt.figure(figsize=(12, 4))
            c1 = (1, 3, 1)
            c2 = (1, 3, 2)
            c3 = (1, 3, 3)
        else:
            fig = plt.figure(figsize=(12, 8))
            c1 = (1, 2, 1)
            c2 = (1, 2, 2)

        fig.add_subplot(*c1)
        plt.scatter(points[:, 0], points[:, 1], c=X[:, k].real)
        plt.colorbar()
        plt.title('X[:,' + str(k) + ']')

        fig.add_subplot(*c2)
        plt.scatter(points[:, 0], points[:, 1], c=Y[:, k].real)
        plt.colorbar()
        plt.title('Y[:,' + str(k) + ']')

        if Y_true is not None:
            fig.add_subplot(*c3)
            plt.scatter(points[:, 0], points[:, 1], c=Y_true[:, k].real)
            plt.colorbar()
            plt.title('Y_true[:,' + str(k) + ']')



def make_D_matrix(q):
    rows = _shared_vars['rows']
    cols = _shared_vars['cols']
    N = _shared_vars['N']
    return sps.coo_matrix((q, (rows, cols)), shape=(N, N), dtype=complex).tocsr()


def D_prime_mode3_contraction(z):
    cols = _shared_vars['cols']
    rows = _shared_vars['rows']
    nnz = _shared_vars['nnz']
    N = _shared_vars['N']
    return sps.coo_matrix((z[cols], (rows, np.arange(nnz))), shape=(N, nnz), dtype=complex).tocsr()


def finite_difference_check_D_prime():
    nnz = _shared_vars['nnz']
    N = _shared_vars['N']
    z = np.random.randn(N)

    q1 = np.random.randn(nnz)
    D1 = make_D_matrix(q1)

    s = 1e-7
    dq = np.random.randn(nnz)
    q2 = q1 + s * dq

    delta = D_prime_mode3_contraction(z) * dq

    D2 = make_D_matrix(q2)

    delta_diff = (D2 * z - D1 * z) / s

    err_D_prime_mode3_contraction = np.linalg.norm(delta - delta_diff) / np.linalg.norm(delta_diff)
    print('s=', s, ', err_D_prime_mode3_contraction=', err_D_prime_mode3_contraction)


def objective(q):
    a_reg = _shared_vars['a_reg']
    X = _shared_vars['X']
    Y = _shared_vars['Y']

    Dx = make_D_matrix(q)
    Res = Y - Dx * X
    Jd = 0.5 * np.linalg.norm(Res) ** 2
    Jr = 0.5 * a_reg * np.linalg.norm(q) ** 2
    return Jd + Jr


def gradient(q):
    a_reg = _shared_vars['a_reg']
    X = _shared_vars['X']
    Y = _shared_vars['Y']
    cols = _shared_vars['cols']
    rows = _shared_vars['rows']

    Dx = make_D_matrix(q)
    Res = Y - Dx * X

    gd = np.zeros(len(q))
    for k in range(len(q)):
        gd[k] = -np.dot(Res[rows[k], :].conj(), X[cols[k], :]).real

    gr = a_reg * q
    return gd + gr


def finite_difference_check_gradient_and_hessian():
    nnz = _shared_vars['nnz']
    H = _shared_vars['H']

    q1 = np.random.randn(nnz)
    J1 = objective(q1)
    g1 = gradient(q1)

    dq = np.random.randn(nnz)
    dJ = np.dot(g1, dq)

    dg = H * dq

    ss = np.logspace(-15, 0, 16)
    errs_grad = np.zeros(len(ss))
    errs_hess = np.zeros(len(ss))

    for k in range(len(ss)):
        s = ss[k]
        q2 = q1 + s * dq

        J2 = objective(q2)
        dJ_diff = (J2 - J1) / s
        err_grad = np.abs(dJ - dJ_diff) / np.abs(dJ_diff)
        errs_grad[k] = err_grad

        g2 = gradient(q2)
        dg_diff = (g2 - g1) / s
        err_hess = np.linalg.norm(dg - dg_diff) / np.linalg.norm(dg_diff)
        errs_hess[k] = err_hess

        print('s=', s, ', err_grad=', err_grad, ', err_hess=', err_hess)

    plt.figure()
    plt.loglog(ss, errs_grad)
    plt.loglog(ss, errs_hess)
    plt.legend(['gradient', 'hessian'])
    plt.xlabel('step size s')
    plt.ylabel('finite difference error')
    plt.title('gradient and hessian finite difference check')









