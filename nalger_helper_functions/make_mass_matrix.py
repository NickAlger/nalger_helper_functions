import dolfin as dl
from .lumping_scheme import *
from .make_fenics_amg_solver import make_fenics_amg_solver


def make_mass_matrix(function_space_V, lumping=None, make_solver=False):
    if lumping is None:
        M = make_mass_matrix_unlumped(function_space_V)
    elif lumping == 'simple':
        M = make_mass_matrix_simple_lumping(function_space_V)
    elif lumping == 'diagonal':
        M = make_mass_matrix_diagonal_lumping(function_space_V)
    elif lumping == 'quadrature':
        M = make_mass_matrix_quadrature_lumping(function_space_V)
    else:
        raise RuntimeError('invalid mass lumping type')

    if make_solver:
        if lumping is not None:
            solve_M = make_fenics_diagonal_matrix_solver(M)
            return M, solve_M
        else:
            solve_M = make_fenics_amg_solver(M)
            return M, solve_M
    else:
        return M

def make_mass_matrix_unlumped(function_space_V):
    V = function_space_V
    u_trial = dl.TrialFunction(V)
    v_test = dl.TestFunction(V)
    mass_form = u_trial * v_test * dl.dx
    M = dl.assemble(mass_form)
    return M

def make_mass_matrix_simple_lumping(function_space_V):
    ML = make_mass_matrix_unlumped(function_space_V)
    ML.zero()
    mass_lumps = dl.assemble(dl.Constant(1.0) * dl.TestFunction(function_space_V) * dl.dx)
    ML.set_diagonal(mass_lumps)
    return ML

def make_mass_matrix_diagonal_lumping(function_space_V):
    # uses diagonal of consistent mass matrix
    ML = make_mass_matrix_unlumped(function_space_V)
    mass_lumps = ML.init_vector(1)
    ML.get_diagonal(mass_lumps)
    ML.zero()
    ML.set_diagonal(mass_lumps)
    return ML

def make_mass_matrix_quadrature_lumping(function_space_V):
    V = function_space_V
    u_trial = dl.TrialFunction(V)
    v_test = dl.TestFunction(V)
    # mass lumping scheme code by Jeremy Bleyer
    # See: https://comet-fenics.readthedocs.io/en/latest/demo/tips_and_tricks/mass_lumping.html
    mass_form = u_trial * v_test * dl.dx(scheme="lumped", degree=2)
    ML = dl.assemble(mass_form)
    return ML

def make_fenics_diagonal_matrix_solver(A):
    idd = A.init_vector(1)
    A.get_diagonal(idd)
    idd[:] = 1.0 / idd[:]
    solve_A = lambda v: idd * v
    return solve_A







