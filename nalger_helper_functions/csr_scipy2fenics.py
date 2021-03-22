import dolfin as dl
from petsc4py import PETSc


def csr_scipy2fenics(A_scipy):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/blob/master/jupyter_notebooks/csr_conversions.ipynb
    A_petsc = PETSc.Mat().createAIJ(size=A_scipy.shape, csr=(A_scipy.indptr, A_scipy.indices, A_scipy.data))
    A_fenics = dl.PETScMatrix(A_petsc)
    return A_fenics