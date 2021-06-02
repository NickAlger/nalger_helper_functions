import dolfin as dl


def make_mass_matrix(function_space_V):
    V = function_space_V
    u_trial = dl.TrialFunction(V)
    v_test = dl.TestFunction(V)
    mass_form = u_trial * v_test * dl.dx
    M = dl.assemble(mass_form)
    return M