import dolfin as dl


def vec2fct(u_vec, Vh):
    u_fct = dl.Function(Vh)
    u_fct.vector()[:] = u_vec.copy()
    return u_fct