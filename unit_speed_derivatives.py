import numpy as np
from math import factorial


run_test = True # requires scipy


def unit_speed_derivatives(YY):
    # Let y(t) be the curve solving the following variable speed ODE:
    #     /y'(t) = v(y(t))
    #     \y(0)  = y0
    # Let x(s) solve the following unit speed version of the ODE:
    #     /x'(s) = v(x(s)) / ||v(x(s))||
    #     \x(0)  = y0
    # Given an array of derivatives of the variable speed path evaluated at 0,
    #     YY:= [y'(0), y''(0), y'''(0), ...],
    # this function computes derivatives of the unit speed path evaluated at 0,
    #     XX:= [x'(0), x''(0), x'''(0), ...].
    # Notice that we can compute XX from YY without knowing the vector field v.
    # Explanation in forthcoming paper:
    #     N. Alger, D.C. Luo, O. Ghattas,
    #     "Higher Order (Gauss-)Newton-Taylor Method for Large-Scale Optimization"
    m = YY.shape[0]
    N = YY.shape[1]-1 # number of derivatives
    ww = np.zeros([m, N+1, N+1]) # zero'th "derivatives"
    for k in range(N+1):
        ww[:,0,k] = YY[:,k]/(np.linalg.norm(YY[:,0])**(k+1))

    for d in range(1,N+1): #first diagonal to Nth diagonal, no zeroth diagonal
        for n in range(1,d+1):
            k = d-n+1
            ww[:,n,k-1] = ww[:,n-1,k] - k*multinom3(ww[:, :n, k-1], ww[:, :n, 0], ww[:, :n, 1])

    XX = ww[:, :, 0]
    return XX


def multinom3(U,V,W):
    n = U.shape[1]-1
    z = np.zeros(U.shape[0])
    zero_dotdot_n = np.arange(n+1)
    for ii in zero_dotdot_n:
        n_minus_ii_dotdot_zero = np.arange(n-ii, -1, -1)
        for jj in n_minus_ii_dotdot_zero:
            kk = n - jj - ii
            c = factorial(n) / (factorial(ii) * factorial(jj) * factorial(kk))
            z = z +  c * U[:,ii] * np.dot(V[:,jj], W[:,kk])
    return z


if run_test:
    from scipy.integrate import solve_ivp
    from scipy import misc


    def v_unit(t, x):
        return np.array([-x[1], x[0]])/np.linalg.norm(x) # circle


    def scaling_factor(x):
        return 1.0 + (x[0] - 0.5)**2 + (x[1] + 0.3)**2 + (x[0] + 0.2)**4 + (x[1] - 0.1)**4 # Randomish positive polynomial


    def v(t, x):
        return v_unit(t, x) * scaling_factor(x)


    y0 = np.array([1,1])/np.sqrt(2.)

    ode_tol = 5e-14
    ode_method = 'RK45'
    finite_difference_step_size = 5e-3
    num_finite_difference_steps = 13

    def y_fct0(T):
        return solve_ivp(v, (0.0, T), y0, rtol=ode_tol, atol=ode_tol, method=ode_method).y[0,-1]

    def y_fct1(T):
        return solve_ivp(v, (0.0, T), y0, rtol=ode_tol, atol=ode_tol, method=ode_method).y[1,-1]

    num_derivatives = 4
    YY = np.zeros((2, num_derivatives))
    for k in range(num_derivatives):
        YY[0,k] = misc.derivative(y_fct0, 0.0, dx=finite_difference_step_size, n=k + 1, order=num_finite_difference_steps)
        YY[1,k] = misc.derivative(y_fct1, 0.0, dx=finite_difference_step_size, n=k + 1, order=num_finite_difference_steps)

    XX = unit_speed_derivatives(YY)

    XX_true = (1. / np.sqrt(2.)) * np.array([[-1, -1,  1, 1, -1, -1,  1, 1, -1, -1,  1, 1],
                                             [ 1, -1, -1, 1,  1, -1, -1, 1,  1, -1, -1, 1]])
    XX_true = XX_true[:,:num_derivatives]

    err = np.linalg.norm(XX - XX_true)
    print('err=', err)
