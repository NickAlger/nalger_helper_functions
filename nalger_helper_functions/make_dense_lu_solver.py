import scipy.linalg as sla


def make_dense_lu_solver(M):
    M_lu, M_pivot = sla.lu_factor(M)
    solve_M = lambda b: sla.lu_solve((M_lu, M_pivot), b)
    return solve_M