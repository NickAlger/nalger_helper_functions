import numpy as np

from nalger_helper_functions import cg_steihaug


# Test cg_steihaug

N = 100
cond = 50

U, _, _ = np.linalg.svd(np.random.randn(N, N))
ss = 1 + np.logspace(0, np.log10(cond), N)
H = U @ np.diag(ss) @ U.T

hessian_matvec = lambda x: H @ x
gradient = np.random.randn(N)
add = lambda u,v: u+v
scale = lambda u,c: c*u
inner_product = lambda u,v: np.dot(u,v)

# Check that this generates the same iterates as reference CG from stackoverflow if:
#  - trust region is infinite,
#  - Hessian is positive definite,
#  - tolerance is zero

max_iter=25

pp = []
callback = lambda z: pp.append(z)

p, aux = cg_steihaug(
    hessian_matvec, gradient,
    trust_region_radius=np.inf, rtol=0.0, max_iter=max_iter, callback=callback,
)


# This basic implementation from stackoverflow agrees with me to machine precision.
# Scipy's CG doesn't seem to agree with me, although the two converge to each other.
# I think this has something to so with where the callback is called in the scipy implementation.
def basic_cg(A, b, x, num_iter):
    # https://stackoverflow.com/a/60847526/484944
    pp_so = []
    r = b - A.dot(x)
    r0 = r
    p = r.copy()
    for i in range(1,num_iter+1):
        Ap = A.dot(p)
        # alpha = np.dot(r, r) / np.dot(p, Ap)
        alpha = np.dot(p, r) / np.dot(p, Ap)
        x = x + alpha * p
        pp_so.append(x)
        r = b - A.dot(x)
        print('i=', i, ', |r|/|r0|=', np.sqrt(np.linalg.norm(r) / np.linalg.norm(r0)))
        beta = -np.dot(r, Ap) / np.dot(p, Ap)
        p = r + beta * p
    return pp_so

pp_so = basic_cg(H, -gradient, -0.0*gradient, max_iter)

err_CG_iterates = np.linalg.norm(np.array(pp) - np.array(pp_so), axis=1) / np.linalg.norm(np.array(pp_so), axis=1)
print('err_CG_iterates=', err_CG_iterates)

# Check that we stop when the tolerance is reached

for rtol in [1e-1, 1e-2, 1e-3]:
    p, aux = cg_steihaug(
        hessian_matvec, gradient,
        trust_region_radius=np.inf, rtol=rtol, max_iter=N,
    )
    relres = np.linalg.norm(H @ p + gradient) / np.linalg.norm(gradient)
    print('rtol=', rtol, ', relres=', relres)

# Check that we stop when we exit the trust region

norm_p_true = np.linalg.norm(np.linalg.solve(H, -gradient))

for scaling in [0.1, 0.5, 0.9, 0.99]:
    trust_radius = scaling * norm_p_true
    p, aux = cg_steihaug(
        hessian_matvec, gradient,
        trust_region_radius=trust_radius, rtol=0.0, max_iter=N,
    )
    norm_p = np.linalg.norm(p)
    print('trust_radius=', trust_radius, ', norm_p=', norm_p)