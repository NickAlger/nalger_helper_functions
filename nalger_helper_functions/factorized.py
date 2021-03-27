import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla


def factorized(A):
    if sps.issparse(A):
        return spla.factorized(A)
    elif isinstance(A, np.ndarray) or isinstance(A, np.matrix):
        return DenseLUFactorization(A)


class DenseLUFactorization:
    def __init__(me, A):
        me.A = A
        me.lu, me.piv = sla.lu_factor(me.A)

    def __call__(me, b): # solve Ax=b
        return sla.lu_solve((me.lu, me.piv), b)