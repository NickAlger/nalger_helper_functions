import fenics
import numpy as np
from fenics_to_scipy_sparse_csr_conversion import convert_fenics_csr_matrix_to_scipy_csr_matrix
import scipy.sparse as sps
import scipy.sparse.linalg as spla

run_test = False

class SmoothBasisMakerNeumann:
    def __init__(me, V, max_smooth_vectors=50):
        R = fenics.FunctionSpace(V.mesh(), 'R', 0)

        u_trial = fenics.TrialFunction(V)
        v_test = fenics.TestFunction(V)
        c_trial = fenics.TrialFunction(R)
        d_test = fenics.TestFunction(R)
        a11 = fenics.inner(fenics.grad(u_trial), fenics.grad(v_test)) * fenics.dx
        a12 = c_trial * v_test * fenics.dx
        a21 = u_trial * d_test * fenics.dx
        A11 = convert_fenics_csr_matrix_to_scipy_csr_matrix(fenics.assemble(a11))
        A12 = convert_fenics_csr_matrix_to_scipy_csr_matrix(fenics.assemble(a12))
        A21 = convert_fenics_csr_matrix_to_scipy_csr_matrix(fenics.assemble(a21))
        me.A = sps.bmat([[A11, A12], [A21, None]]).tocsc()
        solve_A = spla.factorized(me.A)

        m = u_trial * v_test * fenics.dx
        me.M = convert_fenics_csr_matrix_to_scipy_csr_matrix(fenics.assemble(m)).tocsc()
        solve_M = spla.factorized(me.M)

        def solve_neumann(f_vec):
            fe_vec = np.concatenate([f_vec, np.array([0])])
            ue_vec = solve_A(fe_vec)
            u_vec = ue_vec[:-1]
            return u_vec

        me.solve_neumann_linop = spla.LinearOperator((V.dim(), V.dim()), matvec=solve_neumann)
        me.solve_M_linop = spla.LinearOperator((V.dim(), V.dim()), matvec=solve_M)

        ee, UU = spla.eigsh(me.solve_neumann_linop, k=max_smooth_vectors-1, M=me.solve_M_linop, which='LM')

        me.U_smooth = np.zeros((V.dim(), max_smooth_vectors))
        const_fct = np.ones(V.dim())
        me.U_smooth[:,0] = const_fct / np.sqrt(np.dot(const_fct, me.M * const_fct))
        me.U_smooth[:,1:] = solve_M(UU[:,::-1])

        me.k = 0

    def get_smooth_vector(me, k=None):
        if k == None:
            k = me.k
        uk = me.U_smooth[:,k]
        me.k = k + 1
        return uk


if run_test:
    import matplotlib.pyplot as plt
    import mshr

    def vec2fct(u_vec, Vh):
        u_fct = fenics.Function(Vh)
        u_fct.vector()[:] = u_vec.copy()
        return u_fct

    max_smooth_vectors = 15

    outer_circle = mshr.Circle(fenics.Point(0, 0), 3.)
    inner_circle = mshr.Circle(fenics.Point(0, 0), 1.)
    mesh = mshr.generate_mesh(outer_circle - inner_circle, 20)

    V = fenics.FunctionSpace(mesh, 'CG', 1)
    sbm = SmoothBasisMakerNeumann(V, max_smooth_vectors=max_smooth_vectors)

    MIP = np.dot(sbm.U_smooth.T, sbm.M * sbm.U_smooth)
    err_orthogonality = np.linalg.norm(np.eye(max_smooth_vectors) - MIP)
    print('err_orthogonality=', err_orthogonality)

    for k in range(max_smooth_vectors):
        u = sbm.get_smooth_vector()
        plt.figure()
        fenics.plot(vec2fct(u, V), title='smooth function ' + str(k))

