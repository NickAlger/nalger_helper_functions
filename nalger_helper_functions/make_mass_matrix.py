import dolfin as dl
from .lumping_scheme import *

# mass lumping scheme code by Jeremy Bleyer
# See: https://comet-fenics.readthedocs.io/en/latest/demo/tips_and_tricks/mass_lumping.html

def make_mass_matrix(function_space_V, lumped=False):
    V = function_space_V
    u_trial = dl.TrialFunction(V)
    v_test = dl.TestFunction(V)
    if lumped:
        mass_form = u_trial * v_test * dl.dx(scheme="lumped", degree=2)
    else:
        mass_form = u_trial * v_test * dl.dx
    M = dl.assemble(mass_form)
    return M
