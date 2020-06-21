import scipy.sparse as sps
import fenics

perform_tests = False

def convert_fenics_csr_matrix_to_scipy_csr_matrix(A_fenics):
    ai, aj, av = fenics.as_backend_type(A_fenics).mat().getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai))
    return A_scipy

def vec2fct(u_vec, Vh):
    u_fct = fenics.Function(Vh)
    u_fct.vector()[:] = u_vec.copy()
    return u_fct

from petsc4py import PETSc

def convert_scipy_csr_matrix_to_fenics_csr_matrix(A_scipy):
    A_petsc = PETSc.Mat().createAIJ(size=A_scipy.shape, csr=(A_scipy.indptr, A_scipy.indices, A_scipy.data))
    A_fenics = fenics.PETScMatrix(A_petsc)
    return A_fenics


if perform_tests:
    import numpy as np
    from time import time

    # Make Laplacian matrix in fenics
    n = 10
    mesh = fenics.UnitCubeMesh(n,n,n)
    V = fenics.FunctionSpace(mesh, 'CG', 1)

    u = fenics.TrialFunction(V)
    v = fenics.TestFunction(V)
    a = fenics.inner(fenics.grad(u), fenics.grad(v)) * fenics.dx + u * v * fenics.dx

    f = fenics.Function(V)
    f.vector()[:] = np.random.randn(V.dim())
    b = f * v * fenics.dx

    b_fenics = fenics.assemble(b)
    b_numpy = b_fenics[:]

    A_fenics = fenics.assemble(a)


    # Test correctness of matrix converters

    A_scipy = convert_fenics_csr_matrix_to_scipy_csr_matrix(A_fenics)
    A_fenics2 = convert_scipy_csr_matrix_to_fenics_csr_matrix(A_scipy)

    z_fct = fenics.Function(V)
    z_fct.vector().set_local(np.random.randn(V.dim()))
    z = z_fct.vector()
    Az = A_fenics * z
    Az2 = A_fenics2 * z
    err = np.linalg.norm((Az - Az2)[:])/np.linalg.norm(Az[:])
    print('err=', err)


    # Compare timing of fenics solve vs scipy solve
    import scipy.sparse.linalg as spla

    t = time()
    solve_A_scipy = spla.factorized(A_scipy)
    scipy_factorization_time = time() - t
    print('scipy_factorization_time=', scipy_factorization_time)

    t = time()
    x_numpy = solve_A_scipy(b_numpy)
    scipy_solve_time = time() - t
    print('scipy_solve_time=', scipy_solve_time)

    t = time()
    solve_A_fenics = fenics.LUSolver(A_fenics)
    fenics_factorization_time = time() - t
    print('fenics_factorization_time=', fenics_factorization_time)

    x = fenics.Function(V)

    t = time()
    solve_A_fenics.solve(x.vector(), b_fenics)
    fenics_first_solve_time = time() - t
    print('fenics_first_solve_time=', fenics_first_solve_time)

    x2 = fenics.Function(V)

    t = time()
    solve_A_fenics.solve(x2.vector(), b_fenics)
    fenics_second_solve_time = time() - t
    print('fenics_second_solve_time=', fenics_second_solve_time)
