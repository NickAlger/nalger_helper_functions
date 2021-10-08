import dolfin as dl
from .lumping_scheme import *
from .make_fenics_amg_solver import make_fenics_amg_solver

# mass lumping scheme code by Jeremy Bleyer
# See: https://comet-fenics.readthedocs.io/en/latest/demo/tips_and_tricks/mass_lumping.html

def make_mass_matrix(function_space_V, lumped=False, make_solver=False):
    V = function_space_V
    u_trial = dl.TrialFunction(V)
    v_test = dl.TestFunction(V)

    if lumped:
        mass_form = u_trial * v_test * dl.dx(scheme="lumped", degree=2)
    else:
        mass_form = u_trial * v_test * dl.dx

    M = dl.assemble(mass_form)

    if make_solver:
        if lumped:
            mass_lumps = dl.Function(V).vector()
            M.get_diagonal(mass_lumps)
            inv_mass_lumps = dl.Function(V).vector()
            inv_mass_lumps[:] = 1.0 / mass_lumps[:]
            solve_M = lambda v: inv_mass_lumps * v
            return M, solve_M
        else:
            solve_M = make_fenics_amg_solver(M)
            return M, solve_M
    else:
        return M
