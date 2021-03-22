import scipy.sparse as sps
import dolfin as dl


def csr_fenics2scipy(A_fenics):
    # https://github.com/NickAlger/helper_functions/blob/master/csr_conversions.ipynb
    ai, aj, av = dl.as_backend_type(A_fenics).mat().getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai))
    return A_scipy