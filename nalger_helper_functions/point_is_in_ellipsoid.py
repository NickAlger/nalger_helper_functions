import numpy as np


def point_is_in_ellipsoid(p, mu, Sigma, tau):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/jupyter_notebooks
    # True if (p-mu)^T Sigma^-1 (p-mu) <= tau^2
    # p.shape = (N,d) for N points OR p.shape = (d) for one point
    # mu.shape = (d)
    # Sigma.shape = (d,d)
    # tau is a scalar
    # https://github.com/NickAlger/helper_functions/blob/master/point_is_in_ellipsoid.ipynb
    if len(p.shape) == 1:
        p = p.reshape((1,-1))
    N, d = p.shape
    mu = mu.reshape((1,d))
    Sigma = Sigma.reshape((d,d))
    z = (mu - p).T
    return (np.sum(z * np.linalg.solve(Sigma, z), axis=0) <= tau**2).reshape(-1)
