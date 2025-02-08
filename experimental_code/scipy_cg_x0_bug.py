import numpy as np
import scipy.sparse.linalg as spla

# Reference implementation modified from https://stackoverflow.com/a/60847526/484944
def ConjGrad(a, b, x, callback, maxiter, tol):
    r = (b - np.dot(np.array(a), x));
    p = r;
    rsold = np.dot(r.T, r);

    for i in range(maxiter):
        a_p = np.dot(a, p);
        alpha = rsold / np.dot(p.T, a_p);
        x = x + (alpha * p);
        callback(x)
        r = r - (alpha * a_p);
        rsnew = np.dot(r.T, r);
        if (np.sqrt(rsnew) < tol):
            break;
        p = r + ((rsnew / rsold) * p);
        rsold = rsnew;
    return p

# Problem setup

N = 100
cond = 50
maxiter = 25
tol=0.0

U, _, _ = np.linalg.svd(np.random.randn(N, N))
ss = 1 + np.logspace(0, np.log10(cond), N)
A = U @ np.diag(ss) @ U.T
b = np.random.randn(N)


A_linop = spla.LinearOperator((N,N), matvec=lambda x: A @ x)

# Scipy with initial guess unspecified

xx_scipy = []
callback = lambda z: xx_scipy.append(z)
_ = spla.cg(A_linop, b, maxiter=maxiter, tol=tol, callback=callback)

# Scipy with initial guess b

xx_scipy_b = []
callback = lambda z: xx_scipy_b.append(z)
_ = spla.cg(A_linop, b, x0=b, maxiter=maxiter, tol=tol, callback=callback)

# Scipy with initial guess 0

xx_scipy_0 = []
callback = lambda z: xx_scipy_0.append(z)
_ = spla.cg(A_linop, b, x0=np.zeros(N), maxiter=maxiter, tol=tol, callback=callback)

# Reference with initial guess b

xx_ref_b = []
callback = lambda x: xx_ref_b.append(x)
_ = ConjGrad(A, b, b, callback, maxiter, tol)

# Reference with initial guess 0

xx_ref_0 = []
callback = lambda x: xx_ref_0.append(x)
_ = ConjGrad(A, b, np.zeros(N), callback, maxiter, tol)

#

scipy_vs_scipy0 = np.linalg.norm(np.array(xx_scipy)-np.array(xx_scipy_0), axis=1)
print('scipy_vs_scipy0=', scipy_vs_scipy0)

scipy0_vs_scipyb = np.linalg.norm(np.array(xx_scipy_0)-np.array(xx_scipy_b), axis=1)
print('scipy0_vs_scipyb=', scipy0_vs_scipyb)

scipyb_vs_refb = np.linalg.norm(np.array(xx_scipy)-np.array(xx_ref_b), axis=1)
print('scipyb_vs_refb=', scipyb_vs_refb)

scipy0_vs_ref0 = np.linalg.norm(np.array(xx_scipy_b)-np.array(xx_ref_0), axis=1)
print('scipy0_vs_ref0=', scipy0_vs_ref0)

