import numpy as np
from math import factorial


def unit_speed_derivatives(YY):
    # # https://github.com/NickAlger/helper_functions/blob/master/unit_speed_derivatives.py
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