import numpy as np
from scipy.optimize import root_scalar


def multiquadric(rr, e):
    # return e*rr
    return np.sqrt(1. + np.power(e * rr,2))
    # return np.exp(-np.power(e * rr,2))
    # return 1./np.sqrt(1. + np.power(e * rr,2))


def eval_multiquadric_at_points(pp, x0, e):
    rr = np.linalg.norm(x0.reshape((1,-1)) - pp, axis=1)
    return multiquadric(rr, e)


def eval_multiquadric_interpolant_at_points(pp, ww, xx, e):
    # pp = points to evaluate interpolant at
    # ww = rbf weights
    # xx = rbf points
    # e = multiquadric parameter
    ff = np.zeros(pp.shape[0])
    for k in range(len(ww)):
        ff += ww[k] * eval_multiquadric_at_points(pp, xx[k,:], e)
    return ff


def multiquadric_matrix(xx, e):
    # Phi_ij = sqrt(1 + e^2 ||xx[i,:] - xx[j,:]||^2)
    N, d = xx.shape # N=num pts, d=spatial dimension
    Phi = np.zeros((N,N))
    for ii in range(N):
        Phi[ii,:] = eval_multiquadric_at_points(xx, xx[ii,:], e)
    return Phi


def choose_multiquadric_parameter(xx, desired_cond=1e12, e_min=1e-5, e_max=1e5):
    cond_fct = lambda e: np.linalg.cond(multiquadric_matrix(xx, e))

    tau=10
    bracket_max = e_max
    bracket_min = e_max / tau
    while bracket_min > e_min:
        cond = cond_fct(bracket_min)
        print('bracket_min=', bracket_min, ', bracket_max=', bracket_max, ', cond=', cond)
        if cond > desired_cond:
            break
        bracket_max = bracket_min
        bracket_min = bracket_min / tau

    f = lambda log_e: np.log(cond_fct(np.exp(log_e))) - np.log(desired_cond)
    sol = root_scalar(f, bracket=[np.log(bracket_min), np.log(bracket_max)], rtol=1e-2)
    print('sol=', sol)
    e = np.exp(sol.root)
    print('multiquadric e=', e, ', cond_fct(e)=', cond_fct(e))
    return e



####

import matplotlib.pyplot as plt

# N = 85
N = 500
d = 2

xx = np.random.rand(N, d)

# desired_cond=1e12 # 1e6
# e = choose_multiquadric_parameter(xx, desired_cond=desired_cond)
e = 1e3
Phi = multiquadric_matrix(xx, e)

kk=2

ff = np.zeros(N)
ff[kk] = 1.0
# ff = np.ones(N)
ww = np.linalg.solve(Phi, ff)

X, Y = np.meshgrid(np.linspace(-0., 1., 200), np.linspace(-0., 1., 200))
pp = np.vstack([X.reshape(-1), Y.reshape(-1)]).T

zz = eval_multiquadric_interpolant_at_points(pp, ww, xx, e)
Z = zz.reshape(X.shape)

plt.figure()
plt.pcolor(X, Y, Z)
plt.colorbar()
plt.title('e='+str(e))
plt.scatter(xx[:,0], xx[:,1], s=3, c='k')
plt.plot(xx[kk,0], xx[kk,1], '*r')


##

ee = np.logspace(-5,10,20)
cc = np.zeros(ee.shape)
for k in range(len(ee)):
    cc[k] = np.linalg.cond(multiquadric_matrix(xx, ee[k]))

plt.figure()
plt.loglog(ee,cc)