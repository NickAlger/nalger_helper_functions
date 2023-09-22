from __future__ import annotations
import numpy as np
import typing as typ
from collections import deque
# from scipy.optimize import line_search
import scipy.optimize.linesearch as ls
from dataclasses import dataclass
from enum import Enum

VecType = typ.TypeVar('VecType')

def lbfgs(
        cost: typ.Callable[[VecType], float],
        grad: typ.Callable[[VecType], VecType],
        x0: VecType,
        max_vector_pairs_stored: int=20,
        rtol: float=1e-6,
        stag_tol: float=1e-8,
        max_iter: int=100,
        display: bool=True,
        inv_hess0: typ.Union[typ.Callable[[VecType], VecType], LbfgsInverseHessianApproximation]=None,
        num_initial_iter: int = 5 # number of initial iterations before inv_hess0 is used
) -> LbfgsResult:
    '''Computes argmin_x cost(x) via L-BFGS,
    with option for a user-supplied initial inverse Hessian approximation.

    Examples:
    In:
        import numpy as np
        from scipy.optimize import rosen, rosen_der
        from nalger_helper_functions import lbfgs

        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

        print('Minimization with no initial inverse Hessian')
        result = lbfgs(rosen, rosen_der, x0, rtol=1e-10, display=True)
        print('x=', result.x)

        print()
        print('Using previous final inverse Hessian as new initial Hessian after 5 iterations')
        n = len(x0)
        H0 = np.zeros((n,n))
        for k in range(n):
            ek = np.zeros(n)
            ek[k] = 1.0
            H0[:,k] = result.inv_hess.matvec(ek)

        inv_hess0 = lambda x: H0 @ x

        result2 = lbfgs(rosen, rosen_der, x0, inv_hess0=inv_hess0, rtol=1e-10, num_initial_iter=5)
        print('warmstart x=', result2.x)

        print()
        print('Using Scipy L-BFGS implementation for comparison:')
        from scipy.optimize import minimize
        result3 = minimize(rosen, x0, method='L-BFGS-B', jac=rosen_der, tol=1e-10, options={'disp': True})
        print('Scipy x=', result3.x)
    Out:
        Minimization with no initial inverse Hessian
        Iter: 0 , cost: 8.482e+02 , |g|: 2.246e+03 , step_size: 4.497e-04 , using inv_hess0: False
        Iter: 1 , cost: 4.134e+01 , |g|: 2.888e+02 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 2 , cost: 1.683e+01 , |g|: 1.607e+02 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 3 , cost: 2.787e+00 , |g|: 5.671e+01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 4 , cost: 1.045e+00 , |g|: 3.118e+01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 5 , cost: 1.134e-01 , |g|: 1.156e+01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 6 , cost: 6.057e-02 , |g|: 9.535e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 7 , cost: 2.342e-02 , |g|: 1.944e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 8 , cost: 2.218e-02 , |g|: 2.696e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 9 , cost: 2.215e-02 , |g|: 1.352e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 10 , cost: 2.213e-02 , |g|: 1.551e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 11 , cost: 2.206e-02 , |g|: 3.578e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 12 , cost: 2.189e-02 , |g|: 6.885e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 13 , cost: 2.142e-02 , |g|: 1.262e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 14 , cost: 2.029e-02 , |g|: 2.106e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 15 , cost: 1.771e-02 , |g|: 3.206e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 16 , cost: 1.330e-02 , |g|: 4.066e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 17 , cost: 7.709e-03 , |g|: 3.721e+00 , step_size: 3.624e-01 , using inv_hess0: False
        Iter: 18 , cost: 5.483e-03 , |g|: 3.196e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 19 , cost: 3.743e-04 , |g|: 4.732e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 20 , cost: 7.496e-05 , |g|: 3.104e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 21 , cost: 1.830e-05 , |g|: 1.581e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 22 , cost: 4.409e-06 , |g|: 7.752e-02 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 23 , cost: 2.200e-09 , |g|: 2.217e-03 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 24 , cost: 7.297e-12 , |g|: 1.160e-04 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 25 , cost: 9.623e-15 , |g|: 4.032e-06 , step_size: 1.000e+00 , using inv_hess0: False
        LBFGS done.
            Termination reason: RTOL_ACHIEVED
            Iterations: 26
            Cost evaluations: 28
            Gradient evaluations: 28
            Final cost: 3.890703504095597e-18
            Final |g|: 6.66085750413132e-08
        x= [1. 1. 1. 1. 1.]

        Using previous final inverse Hessian as new initial Hessian after 5 iterations
        Iter: 0 , cost: 8.482e+02 , |g|: 2.246e+03 , step_size: 4.497e-04 , using inv_hess0: False
        Iter: 1 , cost: 4.134e+01 , |g|: 2.888e+02 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 2 , cost: 1.683e+01 , |g|: 1.607e+02 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 3 , cost: 2.787e+00 , |g|: 5.671e+01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 4 , cost: 1.045e+00 , |g|: 3.118e+01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 5 , cost: 1.134e-01 , |g|: 1.156e+01 , step_size: 1.000e+00 , using inv_hess0: True
        Iter: 6 , cost: 2.789e-02 , |g|: 6.144e+00 , step_size: 1.000e+00 , using inv_hess0: True
        Iter: 7 , cost: 7.611e-03 , |g|: 2.269e+00 , step_size: 1.000e+00 , using inv_hess0: True
        Iter: 8 , cost: 2.016e-03 , |g|: 9.605e-01 , step_size: 1.000e+00 , using inv_hess0: True
        Iter: 9 , cost: 6.946e-04 , |g|: 8.972e-01 , step_size: 1.000e+00 , using inv_hess0: True
        Iter: 10 , cost: 6.660e-05 , |g|: 3.407e-01 , step_size: 1.000e+00 , using inv_hess0: True
        Iter: 11 , cost: 1.736e-05 , |g|: 1.464e-01 , step_size: 1.000e+00 , using inv_hess0: True
        Iter: 12 , cost: 3.131e-08 , |g|: 5.771e-03 , step_size: 1.000e+00 , using inv_hess0: True
        Iter: 13 , cost: 1.643e-10 , |g|: 4.834e-04 , step_size: 1.000e+00 , using inv_hess0: True
        Iter: 14 , cost: 4.163e-13 , |g|: 2.080e-05 , step_size: 1.000e+00 , using inv_hess0: True
        Iter: 15 , cost: 5.864e-15 , |g|: 2.901e-06 , step_size: 1.000e+00 , using inv_hess0: True
        LBFGS done.
            Termination reason: RTOL_ACHIEVED
            Iterations: 16
            Cost evaluations: 17
            Gradient evaluations: 17
            Final cost: 1.9721444958014692e-19
            Final |g|: 1.4777104336672297e-08
        warmstart x= [1. 1. 1. 1. 1.]

        Using Scipy L-BFGS implementation for comparison:
        RUNNING THE L-BFGS-B CODE
                   * * *
        Machine precision = 2.220D-16
         N =            5     M =           10
        At X0         0 variables are exactly at the bounds
         This problem is unconstrained.
        At iterate    0    f=  8.48220D+02    |proj g|=  2.08540D+03
        At iterate    1    f=  3.99762D+01    |proj g|=  1.69958D+02
        At iterate    2    f=  1.63703D+01    |proj g|=  1.00591D+02
        At iterate    3    f=  2.73198D+00    |proj g|=  4.96538D+01
        At iterate    4    f=  1.01393D+00    |proj g|=  2.20827D+01
        At iterate    5    f=  1.07202D-01    |proj g|=  1.03271D+01
        At iterate    6    f=  2.30119D-02    |proj g|=  8.24714D-01
        At iterate    7    f=  2.22778D-02    |proj g|=  1.74719D-01
        At iterate    8    f=  2.22432D-02    |proj g|=  1.18020D-01
        At iterate    9    f=  2.22175D-02    |proj g|=  1.05946D-01
        At iterate   10    f=  2.21319D-02    |proj g|=  2.88356D-01
        At iterate   11    f=  2.19305D-02    |proj g|=  5.83854D-01
        At iterate   12    f=  2.13934D-02    |proj g|=  1.06513D+00
        At iterate   13    f=  2.00866D-02    |proj g|=  1.76535D+00
        At iterate   14    f=  1.72122D-02    |proj g|=  2.62905D+00
        At iterate   15    f=  1.25070D-02    |proj g|=  3.20973D+00
        At iterate   16    f=  7.12103D-03    |proj g|=  3.34539D+00
        At iterate   17    f=  4.34321D-03    |proj g|=  2.35238D+00
        At iterate   18    f=  2.25392D-04    |proj g|=  4.22986D-01
        At iterate   19    f=  1.96518D-05    |proj g|=  1.19854D-01
        At iterate   20    f=  2.58660D-06    |proj g|=  3.79854D-02
        At iterate   21    f=  1.04796D-07    |proj g|=  1.27803D-02
        At iterate   22    f=  5.45817D-09    |proj g|=  2.12583D-03
        At iterate   23    f=  3.45957D-10    |proj g|=  4.36388D-04
        At iterate   24    f=  1.30034D-12    |proj g|=  1.26799D-05
        At iterate   25    f=  1.91592D-14    |proj g|=  5.08299D-06
                   * * *
        Tit   = total number of iterations
        Tnf   = total number of function evaluations
        Tnint = total number of segments explored during Cauchy searches
        Skip  = number of BFGS updates skipped
        Nact  = number of active bounds at final generalized Cauchy point
        Projg = norm of the final projected gradient
        F     = final function value
                   * * *
           N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
            5     25     27      1     0     0   5.083D-06   1.916D-14
          F =   1.9159228970049549E-014
        CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
        Scipy x= [0.99999999 0.99999999 0.99999998 0.99999996 0.9999999 ]

    In:
        import numpy as np
        from scipy.optimize import rosen, rosen_der
        from nalger_helper_functions import lbfgs

        rosen2 = lambda x: rosen(np.concatenate([x[0], x[1]]))
        def rosen2_der(x):
            der = rosen_der(np.concatenate([x[0], x[1]]))
            return [der[:3], der[3:]]
        x0 = [np.array([1.3, 0.7, 0.8]), np.array([1.9, 1.2])]

        print('Minimization with no initial inverse Hessian')
        result = lbfgs(rosen2, rosen2_der, x0, rtol=1e-10, display=True)
        print('x=', result.x)
    Out:
        Minimization with no initial inverse Hessian
        Iter: 0 , cost: 8.482e+02 , |g|: 2.246e+03 , step_size: 4.497e-04 , using inv_hess0: False
        Iter: 1 , cost: 4.134e+01 , |g|: 2.888e+02 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 2 , cost: 1.683e+01 , |g|: 1.607e+02 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 3 , cost: 2.787e+00 , |g|: 5.671e+01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 4 , cost: 1.045e+00 , |g|: 3.118e+01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 5 , cost: 1.134e-01 , |g|: 1.156e+01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 6 , cost: 6.057e-02 , |g|: 9.535e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 7 , cost: 2.342e-02 , |g|: 1.944e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 8 , cost: 2.218e-02 , |g|: 2.696e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 9 , cost: 2.215e-02 , |g|: 1.352e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 10 , cost: 2.213e-02 , |g|: 1.551e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 11 , cost: 2.206e-02 , |g|: 3.578e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 12 , cost: 2.189e-02 , |g|: 6.885e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 13 , cost: 2.142e-02 , |g|: 1.262e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 14 , cost: 2.029e-02 , |g|: 2.106e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 15 , cost: 1.771e-02 , |g|: 3.206e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 16 , cost: 1.330e-02 , |g|: 4.066e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 17 , cost: 7.709e-03 , |g|: 3.721e+00 , step_size: 3.624e-01 , using inv_hess0: False
        Iter: 18 , cost: 5.483e-03 , |g|: 3.196e+00 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 19 , cost: 3.743e-04 , |g|: 4.732e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 20 , cost: 7.496e-05 , |g|: 3.104e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 21 , cost: 1.830e-05 , |g|: 1.581e-01 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 22 , cost: 4.409e-06 , |g|: 7.752e-02 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 23 , cost: 2.200e-09 , |g|: 2.217e-03 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 24 , cost: 7.297e-12 , |g|: 1.160e-04 , step_size: 1.000e+00 , using inv_hess0: False
        Iter: 25 , cost: 9.623e-15 , |g|: 4.032e-06 , step_size: 1.000e+00 , using inv_hess0: False
        LBFGS done.
            Termination reason: RTOL_ACHIEVED
            Iterations: 26
            Cost evaluations: 28
            Gradient evaluations: 28
            Final cost: 3.890703504095597e-18
            Final |g|: 6.66085750413132e-08
        x= [array([1., 1., 1.]), array([1., 1.])]
    '''
    num_cost_evals: int = 0
    def __cost_with_counter(x):
        nonlocal num_cost_evals
        num_cost_evals += 1
        return cost(x)

    num_grad_evals: int = 0
    last_grad_x: VecType = None
    last_grad_g: VecType = None
    def __grad_with_counter(x):
        nonlocal  num_grad_evals, last_grad_x, last_grad_g
        if last_grad_x is not None:
            if _norm(_sub(last_grad_x, x)) <= 1e-15 * np.max([_norm(x), _norm(last_grad_x)]):
                return last_grad_g

        last_grad_x = x
        last_grad_g = grad(x)
        num_grad_evals += 1
        return last_grad_g

    iter: int = 0
    if isinstance(inv_hess0, LbfgsInverseHessianApproximation) and iter >= num_initial_iter:
        inv_hess = inv_hess0
    elif iter >= num_initial_iter:
        inv_hess = LbfgsInverseHessianApproximation(max_vector_pairs_stored, deque(), deque(), inv_hess0)
    else:
        inv_hess = LbfgsInverseHessianApproximation(max_vector_pairs_stored, deque(), deque(), None)

    x: VecType  = x0
    f: float    = __cost_with_counter(x)
    g: VecType  = __grad_with_counter(x)
    gradnorm: float = _norm(g)

    cost_history: typ.List[float] = [f]
    gradnorm_history: typ.List[float] = [gradnorm]

    p: VecType = inv_hess.matvec(_neg(g))
    if inv_hess.inv_hess0 is None:
        old_old_fval = _add(f, _norm(g) / 2.0)
    else:
        old_old_fval = None
    line_search_result = _line_search(__cost_with_counter, __grad_with_counter, x, p, g, f, old_old_fval)
    step_size: float = line_search_result[0]

    step_size_history: typ.List[float] = [step_size]

    def __display_iter_info():
        if display:
            print('Iter:', iter,
                  ', cost:', np.format_float_scientific(f, precision=3, unique=False),
                  ', |g|:', np.format_float_scientific(gradnorm, precision=3, unique=False),
                  ', step_size:', None if step_size is None else np.format_float_scientific(step_size, precision=3, unique=False),
                  ', using inv_hess0:', inv_hess.inv_hess0 is not None)

    __display_iter_info()
    f0: float = f
    gradnorm0: float = gradnorm

    termination_reason: LbfgsTerminationReason = LbfgsTerminationReason.MAXITER_REACHED
    while iter < max_iter:
        if step_size is None:
            termination_reason = LbfgsTerminationReason.LINESEARCH_FAILED
            break

        iter += 1
        if iter == num_initial_iter:
            if isinstance(inv_hess0, LbfgsInverseHessianApproximation):
                inv_hess = inv_hess0
            else:
                inv_hess = LbfgsInverseHessianApproximation(max_vector_pairs_stored, deque(), deque(), inv_hess0)

        x_old = x
        f_old = f
        g_old = g

        x = _add(x, _componentwise_scalar_mult(p, step_size)) # x + step_size * p
        f = line_search_result[1]
        g = __grad_with_counter(x)
        gradnorm = _norm(g)

        cost_history.append(f)
        gradnorm_history.append(gradnorm)

        if gradnorm <= rtol * gradnorm0:
            termination_reason: LbfgsTerminationReason = LbfgsTerminationReason.RTOL_ACHIEVED
            break

        if (f_old - f) < stag_tol * f_old:
            termination_reason: LbfgsTerminationReason = LbfgsTerminationReason.DESCENT_STAGNATED
            break

        inv_hess.add_new_s_y_pair(_sub(x, x_old), _sub(g, g_old)) # s = x - x_old, y = g - g_old
        p = inv_hess.matvec(_neg(g))
        line_search_result = _line_search(__cost_with_counter, __grad_with_counter, x, p, g, f, None)
        step_size = line_search_result[0]

        step_size_history.append(step_size)

        __display_iter_info()

    if display:
        print('LBFGS done.')
        print('    Termination reason:', termination_reason.name)
        print('    Iterations:', iter)
        print('    Cost evaluations:', num_cost_evals)
        print('    Gradient evaluations:', num_grad_evals)
        print('    Final cost:', f)
        print('    Final |g|:', gradnorm)

    return LbfgsResult(x, f, g, inv_hess,
                       iter, num_cost_evals, num_grad_evals,
                       cost_history, gradnorm_history, step_size_history,
                       termination_reason)


class LbfgsTerminationReason(Enum):
    MAXITER_REACHED = 0
    RTOL_ACHIEVED = 1
    DESCENT_STAGNATED = 2
    LINESEARCH_FAILED = 3


class LbfgsResult(typ.NamedTuple):
    x: VecType       # solution
    cost: float         # cost(x)
    grad: VecType    # grad f(x)
    inv_hess: LbfgsInverseHessianApproximation   # L-BFGS approximation to the inverse Hessian at x
    iter: int
    num_cost_evals: int
    num_grad_evals: int
    cost_history: typ.List[float]      # [cost(x0), cost(x1), ..., cost(x)]
    gradnorm_history: typ.List[float]      # [grad(x0), grad(x1), ..., grad(x)]
    step_size_history: typ.List[float] # [a0, a1, ...], where x1 = x0 + a0*p, x2 = x1 + a1*p, ...
    termination_reason: LbfgsTerminationReason


@dataclass
class LbfgsInverseHessianApproximation:
    '''See Nocedal and Wright page 177-179.'''
    m: int # max vector pairs stored
    ss: typ.Deque[VecType] # GETS MODIFIED! ss=[s_(k-1), s_(k-2), ..., s_(k-m)], s_i = x_(i+1) - x_i. Eq 7.18, left, on page 177
    yy: typ.Deque[VecType] # GETS MODIFIED! yy=[y_(k-1), y_(k-2), ..., y_(k-m)], y_i = grad f_(i+1) - grad f_k. Eq 7.18, right, on page 177
    inv_hess0: typ.Callable[[VecType], VecType] = None # Initial inverse Hessian approximation

    def __post_init__(me) -> None:
        assert(me.m >= 0)
        assert(len(me.ss) == len(me.yy))
        while len(me.ss) > me.m:
            me.ss.pop()
        while len(me.yy) > me.m:
            me.yy.pop()

    def add_new_s_y_pair(
            me,
            s: VecType,
            y: VecType
    ) -> None:
        me.ss.appendleft(s)
        if len(me.ss) > me.m:
            me.ss.pop()

        me.yy.appendleft(y)
        if len(me.yy) > me.m:
            me.yy.pop()

    def apply_inv_hess0_k(me, x: VecType) -> VecType:
        if me.inv_hess0 is not None:
            return me.inv_hess0(x)
        else:
            if me.ss:
                gamma_k = _inner_product(me.ss[0], me.yy[0]) / _inner_product(me.yy[0], me.yy[0]) # <s_(k-1), y_(k-1)> / <y_(k-1), y_(k-1)>
            else:
                gamma_k = 1.0
            return _componentwise_scalar_mult(x, gamma_k) # H0_k = gamma_k*I

    def matvec(me, q: VecType) -> VecType:
        '''Computes
            r = H_k grad f_k
        via L-BFGS two-loop recursion.
        Algorithm 7.4 on page 178 of Nocedal and Wright.
        '''
        rhos = [_componentwise_inverse(_inner_product(y, s)) for s, y in zip(me.ss, me.yy)] # 1.0 / inner(y, s). equation 7.17 (left) on page 177
        alphas = []
        for s, y, rho in zip(me.ss, me.yy, rhos):
            alpha = rho * _inner_product(s, q)
            q = _sub(q, _componentwise_scalar_mult(y, alpha)) # q = q - alpha*y
            alphas.append(alpha)
        r = me.apply_inv_hess0_k(q)
        for s, y, rho, alpha in zip(reversed(me.ss), reversed(me.yy), reversed(rhos), reversed(alphas)):
            beta = rho * _inner_product(y, r)
            r = _add(r, _componentwise_scalar_mult(s, alpha - beta)) # r = r + s * (alpha - beta)
        return r


def _is_container(x):
    return isinstance(x, typ.Iterable) and (not isinstance(x, np.ndarray))

def _inner_product(x, y) -> float:
    if _is_container(x):
        return np.sum([_inner_product(xi, yi) for xi, yi in zip(x, y)])
    else:
        return (x * y).sum()

def _norm(x) -> float: # ||x||
    return np.sqrt(_inner_product(x, x))

def _add(x, y): # x + y
    if _is_container(x):
        T = type(x)
        return T([_add(xi, yi) for xi, yi in zip(x, y)])
    else:
        return x + y

def _neg(x): # -x
    if _is_container(x):
        T = type(x)
        return T([_neg(xi) for xi in x])
    else:
        return -x

def _sub(x, y): # x - y
    return _add(x, _neg(y))

def _componentwise_scalar_mult(x, s):  # x * s. x is a vector and s is a scalar
    if _is_container(x):
        T = type(x)
        return T([_componentwise_scalar_mult(xi, s) for xi in x])
    else:
        return x * s

def _componentwise_inverse(x):
    if _is_container(x):
        T = type(x)
        return T([_componentwise_inverse(xi) for xi in zip(x)])
    else:
        return 1.0 / x

def _line_search(cost, grad, x, p, g, f, old_old_fval, **kwargs):
    def cost_1D(s):
        return cost(_add(x, _componentwise_scalar_mult(p, s)))

    def grad_1D(s):
        return _inner_product(p, grad(_add(x, _componentwise_scalar_mult(p, s))))

    g0 = _inner_product(p, g)
    stp, fval, old_fval = ls.scalar_search_wolfe1(cost_1D, grad_1D, f, old_old_fval, g0, **kwargs)
    return stp, fval
