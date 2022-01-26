import numpy as np

# WORK IN PROGRESS. DO NOT USE UNLESS YOU KNOW WHAT YOU ARE DOING

def generalized_leapfrog_integrator(final_time_T : float,
                                    num_timesteps_N : int,
                                    hamiltonian_object,
                                    omega=1.0,
                                    save_path=False,
                                    hamiltonian_tol=5e-2,
                                    try_harder_upon_failure=True,
                                    verbose=False):
    # See:
    # Tao, Molei. "Explicit symplectic approximation of nonseparable Hamiltonians: algorithm and long time performance"
    # https://arxiv.org/abs/1609.02212
    # and
    # Cobb, Adam et. al. "Introducing an Explicit Symplectic Integration Scheme for Riemannian Manifold Hamiltonian Monte Carlo"
    # https://arxiv.org/abs/1910.06243
    #
    # hamiltonian_object must implement the following methods:
    #   hamiltonian_object.position()        : None     -> n-vector
    #   hamiltonian_object.momentum()        : None     -> n-vector
    #   hamiltonian_object.update_position() : n-vector -> None
    #   hamiltonian_object.update_momentum() : n-vector -> None
    #   hamiltonian_object.hamiltonian()     : None     -> n-vector
    #   hamiltonian_object.dH_dx()           : None     -> n-vector
    #   hamiltonian_object.dH_dp()           : None     -> n-vector
    #   hamiltonian_object.copy()            : None     -> another hamiltonian object
    #
    # Here an "n-vector" is any implementation of a mathematical vector in R^n (such as a numpy array)
    # that can be added to another n-vector via __add__(), i.e., u+v,
    # and scaled by a float via __mul__() methods, i.e., a*u

    Ha = hamiltonian_object        # Ha: w', p
    Hb = hamiltonian_object.copy() # Hb: w, p'

    w0 = Ha.position()
    p0 = Ha.momentum()
    ww = []
    pp = []

    initial_hamiltonian = Ha.hamiltonian()

    integration_failed = False
    dt = float(final_time_T) / float(num_timesteps_N)

    _phi_a(dt / 2., Ha, Hb) # initial half-step
    for k in range(num_timesteps_N):
        if verbose:
            print('k=', k, ', Ha=', Ha.hamiltonian(), ', Hb=', Hb.hamiltonian(),
                  ', ||x-x0||=', np.linalg.norm(Ha.position() - w0),
                  ', ||p||=',    np.linalg.norm(Ha.momentum()))

        rel_err1 = np.abs(Ha.hamiltonian() - initial_hamiltonian) / np.abs(initial_hamiltonian)
        rel_err2 = np.abs(Hb.hamiltonian() - initial_hamiltonian) / np.abs(initial_hamiltonian)
        if (rel_err1 > hamiltonian_tol) or (rel_err2 > hamiltonian_tol):
            print('Integration failed: Hamiltonian changed too much')
            integration_failed = True
            break

        _phi_b(dt / 2., Ha, Hb)
        _phi_c(dt,      Ha, Hb, omega)
        _phi_b(dt / 2., Ha, Hb)
        if k < (num_timesteps_N - 1):
            _phi_a(dt, Ha, Hb)

        if save_path:
            ww.append(Hb.position())
            pp.append(Ha.momentum())

    _phi_a(dt / 2., Ha, Hb) # final half-step

    hamiltonian_object.update_position(0.5*(Ha.position()+Hb.position()))
    hamiltonian_object.update_momentum(0.5*(Ha.momentum()+Hb.momentum()))

    if integration_failed and try_harder_upon_failure:
        print('Trying again with twice as many timesteps.')
        hamiltonian_object.update_position(w0)
        hamiltonian_object.update_momentum(p0)
        return generalized_leapfrog_integrator(final_time_T, 2*num_timesteps_N, hamiltonian_object,
                                               omega=omega, save_path=save_path, hamiltonian_tol=hamiltonian_tol,
                                               try_harder_upon_failure=True, verbose=verbose)

    if save_path:
        return np.array(ww), np.array(pp)

def _coupling_hamiltonian(Ha, Hb):
    position_residual = 0.5*np.linalg.norm(Ha.position() - Hb.position())**2
    momentum_residual = 0.5*np.linalg.norm(Ha.momentum() - Hb.momentum())**2
    return position_residual + momentum_residual

def _R(dt, x_diff, p_diff, omega):
    R_x =   np.cos(2. * omega * dt) * x_diff + np.sin(2. * omega * dt) * p_diff
    R_p = - np.sin(2. * omega * dt) * x_diff + np.cos(2. * omega * dt) * p_diff
    return R_x, R_p

def _phi_a(dt, Ha, Hb):
    Ha.update_position(Ha.position() + Hb.dH_dp() * dt) # w' <- w' + dt * H(w,p')
    Ha.update_momentum(Ha.momentum() - Hb.dH_dx() * dt) # p  <- p  - dt * H(w,p')

def _phi_b(dt, Ha, Hb):
    Hb.update_position(Hb.position() + Ha.dH_dp() * dt) # w  <- w  + dt * H(w',p)
    Hb.update_momentum(Hb.momentum() - Ha.dH_dx() * dt) # p' <- p' - dt * H(w',p)

def _phi_c(dt, Ha, Hb, omega):
    x_sum  = Hb.position() + Ha.position() # w + w'
    x_diff = Hb.position() - Ha.position() # w - w'
    p_sum  = Ha.momentum() + Hb.momentum() # p + p'
    p_diff = Ha.momentum() - Hb.momentum() # p - p'

    Rx_diff, Rp_diff = _R(dt, x_diff, p_diff, omega)
    Hb.update_position(0.5 * (x_sum + Rx_diff)) # update w
    Ha.update_momentum(0.5 * (p_sum + Rp_diff)) # update p
    Ha.update_position(0.5 * (x_sum - Rx_diff)) # update w'
    Hb.update_momentum(0.5 * (p_sum - Rp_diff)) # update p'