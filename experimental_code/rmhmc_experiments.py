import numpy as np
import typing as typ
import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass


from jax import config
config.update("jax_enable_x64", True)  # for finite difference checks

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# See:
# Brofos, James, and Roy R. Lederman.
# "Evaluating the implicit midpoint integrator for riemannian hamiltonian monte carlo."
# International Conference on Machine Learning. PMLR, 2021.

def fixed_point_iteration(
    x0: jnp.ndarray,
    f: typ.Callable[[jnp.ndarray], jnp.ndarray],
    rtol: float = 1e-3,
    maxiter: int = 30,
) -> jnp.ndarray: # solution of z=f(z)
    '''Solves x=f(x)
    See Algorithm 1 in Brofos, James, and Roy R. Lederman
    '''
    dx = np.inf
    xp = x0
    for ii in range(maxiter):
        xpp = f(xp)
        if ii == 0:
            norm_f0 = np.linalg.norm(xpp)
        if dx <= rtol * norm_f0:
            break
        dx = jnp.max(jnp.abs(xpp - xp))
        xp = xpp
    return xp


def implicit_midpoint_step(
        position: jnp.ndarray, # position vector, shape=(N,)
        momentum: jnp.ndarray, # momentum vector, shape=(N,)
        H_and_dH: typ.Callable[
            [jnp.ndarray, jnp.ndarray], # (q, p)
            typ.Tuple[
                jnp.ndarray, # H, shape=(N,)
                typ.Tuple[jnp.ndarray, jnp.ndarray]] # (dH/dq, dH/dp), shapes=(N,) for each
        ], # (q,p) -> (H, (dH/dq, dH/dp))
        step_size:  float,          # epsilon from paper
        rtol:       float   = 1e-6, # tolerance (delta) for implicit midpoint solve
        maxiter:    int     = 30,
        display:    bool    = False,
        check_hamiltonian: bool = True,
) -> typ.Tuple[
    jnp.ndarray, # new_q, shape=(N,)
    jnp.ndarray, # new_p, shape=(N,)
]:
    '''Takes one step of the implicit midpoint method for Hamiltoniam ODE integration.
    See Algorithm 3 in Brofos, James, and Roy R. Lederman
    '''
    N = len(position)
    assert(position.shape == (N,))
    assert(momentum.shape == (N,))

    num_dH_evals = [0] # counter hack
    def f(
        z: jnp.ndarray, # (q_prime, p_prime). shape=(2N,)
    ) -> jnp.ndarray: # (del H / del p, -del H / del q), shape=(2N,)
        num_dH_evals[0] += 1
        qbar = 0.5 * (z[:N] + position)
        pbar = 0.5 * (z[N:] + momentum)
        _, (dH_dq_zbar, dH_dp_zbar) = H_and_dH(qbar, pbar)
        fq = position + step_size * dH_dp_zbar
        fp = momentum - step_size * dH_dq_zbar
        return jnp.concatenate([fq, fp])

    z0 = jnp.concatenate([position, momentum])
    if check_hamiltonian:
        H_old, _ = H_and_dH(position, momentum)

    z = fixed_point_iteration(z0, f, rtol, maxiter=maxiter)

    new_q = z[:N]
    new_p = z[N:]
    if check_hamiltonian:
        H_new, _ = H_and_dH(new_q, new_p)

    if display:
        if check_hamiltonian:
            print('H_old=', H_old, ', H_new=', H_new, ', num_del_H_evals[0]=', num_dH_evals[0])
        else:
            print('num_dH_evals[0]=', num_dH_evals[0])

    return new_q, new_p


def implicit_midpoint_integration(
        initial_position: jnp.ndarray, # initial position q, shape=(N,)
        initial_momentum: jnp.ndarray, # initial momentum p, shape=(N,)
        hamiltonian_and_hamiltonian_gradient: typ.Callable[
            [jnp.ndarray, jnp.ndarray], # (position_q, momentum_p), shapes=(N,) for each
            typ.Tuple[
                jnp.ndarray,                            # H, scalar, shape=()
                typ.Tuple[jnp.ndarray, jnp.ndarray]],   # (dH/dq, dH/dp), shapes=(N,) for each
        ],
        integration_time:   float = 1.0,
        num_steps:          int   = 10,
        solver_rtol:        float = 1e-6, # tolerance (delta) for implicit midpoint solve
        solver_maxiter:     int   = 30,
        display:            bool  = False,
        return_intermediate_steps: bool = False,
) -> typ.Tuple[
    jnp.ndarray, # new_q, shape=(N,)
    jnp.ndarray, # new_p, shape=(N,)
]:
    '''Integrates Hamilton's ODE via implicit midpoint method.
    See Algorithm 3 in Brofos, James, and Roy R. Lederman
    '''

    step_size = integration_time / num_steps

    qq = [initial_position.copy()]
    pp = [initial_momentum.copy()]
    def _update(new_q, new_p):
        if return_intermediate_steps:
            qq.append(new_q)
            pp.append(new_p)
        else:
            qq[0] = new_q
            pp[0] = new_p

    for k in range(num_steps):
        q, p = implicit_midpoint_step(
            qq[-1], pp[-1], hamiltonian_and_hamiltonian_gradient, step_size,
            display=display, rtol=solver_rtol, maxiter=solver_maxiter)
        _update(q, p)

    if return_intermediate_steps:
        return jnp.array(qq),jnp.array(pp)
    else:
        return qq[0], pp[0]


#

@jax.jit
def least_squares_rmhmc_hamiltonian_and_hamiltonian_gradient(
        q: jnp.ndarray, # position, shape=(N,)
        p: jnp.ndarray, # momentum, shape=(N,)
        y: jnp.ndarray, # data, shape=(M,)
        f: float,       # f(q),                shape=(M,)
        F: jnp.ndarray, # F = df(q) / dq,      shape=(M,N)
        S: np.ndarray,  # S = d^2f(q) / dq^2,  shape=(M,N,N)
        iSigma: jnp.ndarray,  # noise precision matrix, shape=(M,M)
        prior_mean: jnp.ndarray, # shape=(N,)
        iC: jnp.ndarray, # prior precision matrix, shape=(N,N)
):
    '''Hamiltonian and Hamiltonian gradient for RMHMC sampling from exp(-J(q)), where
    J(q) = 1/2 ||y - f(q)||_Sigma^2 + 1/2 ||q - q0||_C^2

    See TensorHMC document for details
    '''
    res_d = y - f # data residual
    res_r = prior_mean - q # prior residual

    J_d = 0.5 * res_d.T @ (iSigma @ res_d)
    J_r = 0.5 * res_r.T @ (iC @ res_r)
    J = J_d + J_r # negative log posterior

    g_d = -F.T @ (iSigma @ res_d)
    g_r = -iC @ res_r
    g = g_d + g_r # gradient of negative log posterior

    G = F.T @ (iSigma @ F) + iC # metric

    ee, P = jnp.linalg.eigh(G)
    z = P.T @ p
    H = J + 0.5*jnp.sum(jnp.log(ee)) + 0.5*jnp.sum(z**2 / ee) # Hamiltonian: J + log(sqrt(det(G)) + 0.5*p.T @ inv(G) @ p

    Psi = iSigma @ (F @ (P @ (jnp.diag(1.0 / ee) @ P.T))) # iSigma @ F @ inv(G), shape=(M,N)
    eta = P @ (z / ee) # inv(G) @ p, shape=(N,)
    xi = iSigma @ (F @ eta) # shape=(M,)

    dH_dp = eta
    dH_dq = g + jnp.einsum('xij,xi->j', S, Psi - jnp.outer(xi, eta))

    return H, (dH_dq, dH_dp)


def least_squares_rmhmc_proposal(
        current_q: jnp.ndarray, # shape=(N,)
        y: jnp.ndarray, # data, shape=(M,)
        compute_f_and_derivatives: typ.Callable[
            [jnp.ndarray], # q, shape=(N,)
            typ.Tuple[
                jnp.ndarray, # f(q),                shape=(M,)
                jnp.ndarray, # F = df(q) / dq,      shape=(M,N)
                jnp.ndarray, # S = d^2f(q) / dq^2,  shape=(M,N,N)
            ]
        ],
        iSigma: jnp.ndarray,  # noise precision matrix, shape=(M,M)
        prior_mean: jnp.ndarray, # shape=(N,)
        iC: jnp.ndarray, # prior precision matrix, shape=(N,N)
        integration_time:   float   = 1.0,
        num_steps:          int     = 10,
        integrator_options: dict = None
):
    '''Proposed sample from exp(-J(q)), where
    J(q) = 1/2 ||y - f(q)||_Sigma^2 + 1/2 ||q - q0||_C^2

    See TensorHMC document for details
    '''
    integrator_options = {} if integrator_options is None else integrator_options
    N = len(current_q)

    def H_and_dH(q, p):
        f, F, S = compute_f_and_derivatives(q)
        return least_squares_rmhmc_hamiltonian_and_hamiltonian_gradient(
            q, p, y, f, F, S, iSigma, prior_mean, iC
        )

    _, F, _ = compute_f_and_derivatives(current_q)
    G = F.T @ (iSigma @ F) + iC # metric
    ee, P = jnp.linalg.eigh(G)
    # current_p = (P.T @ jnp.array(np.random.randn(N)))
    current_p = P @ ((P.T @ jnp.array(np.random.randn(N))) * jnp.sqrt(ee)) # <-- should be the right one
    # current_p = (P @ jnp.array(np.random.randn(N))) * jnp.sqrt(ee)

    proposed_q, proposed_p = implicit_midpoint_integration(
        current_q, current_p, H_and_dH, integration_time=integration_time, num_steps=num_steps, **integrator_options
    )
    return proposed_q, proposed_p




####

H = lambda q, p: 0.5 * jnp.dot(q, q) + 0.5 * jnp.dot(p, p)
dH = lambda q, p: (q, p)
H_and_dH = lambda q, p: (H(q,p), dH(q, p))

q0 = jnp.array([1.0, 0.0])
p0 = jnp.array([0.0, 1.0])

integration_time = 20.0
num_steps = 20

qq, pp = implicit_midpoint_integration(
    q0, p0, H_and_dH, integration_time=integration_time, num_steps=num_steps,
    solver_rtol=1e-8, display=True,
    return_intermediate_steps=True)


plt.plot(qq[:,0], qq[:,1])

####

@jax.jit
def G(q): # metric
    G00 = 1.1 + jnp.sin(q[0])
    G11 = 1.1 + jnp.cos(q[1])
    return jnp.array([[G00, 0.0], [0.0, G11]])

H = jax.jit(lambda q, p: 0.5 * jnp.dot(q, q) + 0.5 * jnp.dot(p, G(q) @ p))
dH = jax.jit(jax.jacfwd(H, argnums=[0,1]))
H_and_dH = lambda q, p: (H(q,p), dH(q,p))

q0 = jnp.array([1.0, 0.0])
p0 = jnp.array([0.0, 1.0])

integration_time = 20.0
num_steps = 100
rtol=1e-6
method= 'fixed_point' # 'broyden1'
solver_maxiter=10

qq, pp = implicit_midpoint_integration(
    q0, p0, H_and_dH, integration_time=integration_time, num_steps=num_steps,
    solver_rtol=rtol, display=True, solver_maxiter=solver_maxiter,
    return_intermediate_steps=True)

qq_reverse, pp_reverse = implicit_midpoint_integration(
    qq[-1,:], -pp[-1,:], H_and_dH, integration_time=integration_time, num_steps=num_steps,
    solver_rtol=rtol, display=True, solver_maxiter=solver_maxiter,
    return_intermediate_steps=True)

plt.figure()
plt.plot(qq[:,0], qq[:,1], 'k')
plt.plot(qq_reverse[:,0], qq_reverse[:,1], '--r')
plt.legend(['Forward integration', 'Reverse integration'])
plt.title('Implicit midpoint integration,\n #fixed point iter=' + str(solver_maxiter) + '\n#steps='+str(num_steps))
plt.savefig('implicit_midpoint_iter='+str(solver_maxiter)+'_steps' + str(num_steps) + '.png', bbox_inches='tight', dpi=300)

q0_2 = qq_reverse[-1,:]
p0_2 = pp_reverse[-1,:]

err_reverse_q = np.linalg.norm(q0_2 - q0) / np.linalg.norm(q0)
err_reverse_p = np.linalg.norm(p0_2 + p0) / np.linalg.norm(p0)
print('err_reverse_q=', err_reverse_q)
print('err_reverse_p=', err_reverse_p)

####

a = 1
b = 100

prior_parameter = 2.0 # <-- Experiment with this
likelihood_parameter = 1.0 # <--- Experiment with this

y = jnp.array([a, 0])
iSigma = likelihood_parameter*jnp.eye(2)
iC = prior_parameter*jnp.eye(2)
prior_mean = jnp.array([0.0, 0.0])

def f(q):
    return jnp.array([q[0], jnp.sqrt(b) * (q[1] - q[0]**2)])

def J_d(q):
    res_d = y - f(q)
    return 0.5 * res_d.T @ (iSigma @ res_d)

def J_r(q):
    res_r = prior_mean - q
    return 0.5 * res_r.T @ (iC @ res_r)

def J(q):
    return J_d(q) + J_r(q)

def pi_post(q):
    return jnp.exp(-J(q))

xmin = -2.0
xmax = 2.0
ymin = -1.0
ymax = 3.0
nx=100
ny=nx

xx = np.linspace(xmin, xmax, nx)
yy = np.linspace(ymin, ymax, ny)

X, Y = np.meshgrid(xx, yy)
Post = np.zeros(X.shape)
for ii in range(X.shape[0]):
    for jj in range(X.shape[1]):
        Post[ii,jj] = pi_post(jnp.array([X[ii,jj], Y[ii,jj]]))


df = jax.jacfwd(f)
ddf = jax.jacfwd(df)

compute_f_and_derivatives = jax.jit(lambda q: (f(q), df(q), ddf(q)))


# finite difference check

def H_and_dH(q, p):
    f, F, S = compute_f_and_derivatives(q)
    return least_squares_rmhmc_hamiltonian_and_hamiltonian_gradient(
        q, p, y, f, F, S, iSigma, prior_mean, iC
    )

q1 = jnp.array(np.random.randn(2))
p1 = jnp.array(np.random.randn(2))
H1, (dH_dq, dH_dp) = H_and_dH(q1, p1)
s = 1e-6
dq = jnp.array(np.random.randn(2))
dp = jnp.array(np.random.randn(2))
q2 = q1 + s * dq
p2 = p1 + s * dp

dHq = jnp.dot(dH_dq, dq)
dHp = jnp.dot(dH_dp, dp)

H2q, _ = H_and_dH(q2, p1)
dHq_diff = (H2q - H1) / s
err_dHq = np.linalg.norm(dHq - dHq_diff) / np.linalg.norm(dHq_diff)

H2p, _ = H_and_dH(q1, p2)
dHp_diff = (H2p - H1) / s
err_dHp = np.linalg.norm(dHp - dHp_diff) / np.linalg.norm(dHp_diff)

print('s=', s, ', err_dHq=', err_dHq, ', err_dHp=', err_dHp)

#
max_samples=100
current_q = jnp.array([0.0, 0.25])
qqq = []
for _ in tqdm(range(max_samples)):
    qq, pp = least_squares_rmhmc_proposal(
        current_q, y, compute_f_and_derivatives, iSigma, prior_mean, iC,
        integration_time=1.0, num_steps=10,
        integrator_options={'return_intermediate_steps':True,
                            'display':False, 'solver_rtol':1e-3}
    )
    qqq.append(qq.copy())
    current_q = qq[-1,:]

samples = jnp.array([qq[-1,:] for qq in qqq])

plt.figure()
for num_samples in range(2,len(qqq)+1):
# for num_samples in range(1,10):
    plt.clf()
    # plt.imshow(Post)
    plt.pcolor(X, Y, Post)
    # plt.colorbar()
    # for qq in qqq[:num_samples-1]:
    #     plt.plot(qq[:,0], qq[:,1], 'gray')
    plt.plot(qqq[num_samples-1][:, 0], qqq[num_samples-1][:, 1], 'r')
    plt.plot(samples[:num_samples-1,0], samples[:num_samples-1,1], '.', color='k', markersize=5)
    plt.plot(samples[num_samples-2, 0], samples[num_samples-2, 1], '.', color='k', markersize=10)
    plt.plot(samples[num_samples-1, 0], samples[num_samples-1, 1], '.', color='r', markersize=10)
    plt.xlim(-1.25, 1.5)
    plt.ylim(-0.5, 1.75)
    plt.savefig('rmhmc_samples_with_last_trajectory'+str(num_samples)+'.png', bbox_inches='tight', dpi=300)


plt.figure()
plt.plot(samples[:,0])