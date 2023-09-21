from __future__ import annotations
import numpy as np
import typing as typ
from collections import deque
from scipy.optimize import line_search
# import scipy.optimize.linesearch as ls
from dataclasses import dataclass
from enum import Enum


def lbfgs(
        cost: typ.Callable[[np.ndarray], float],
        grad: typ.Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        max_vector_pairs_stored: int=20,
        rtol: float=1e-6,
        stag_tol: float=1e-8,
        max_iter: int=100,
        display: bool=True,
        inv_hess0: typ.Union[typ.Callable[[np.ndarray], np.ndarray], LbfgsInverseHessianApproximation]=None,
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
        Iter: 0 , cost: 848.22 , |g|_inf: 2085.4 , step_size: 0.0004496668508702465 , using inv_hess0: False
        Iter: 1 , cost: 41.34323740111179 , |g|_inf: 178.40189326409805 , step_size: 1.0 , using inv_hess0: False
        Iter: 2 , cost: 16.826849424945813 , |g|_inf: 105.5379767341664 , step_size: 1.0 , using inv_hess0: False
        Iter: 3 , cost: 2.7868248969070386 , |g|_inf: 51.01396443585729 , step_size: 1.0 , using inv_hess0: False
        Iter: 4 , cost: 1.0446811105748832 , |g|_inf: 21.4422281910189 , step_size: 1.0 , using inv_hess0: False
        Iter: 5 , cost: 0.11343722694575001 , |g|_inf: 10.271531675646154 , step_size: 1.0 , using inv_hess0: False
        Iter: 6 , cost: 0.060566823074288886 , |g|_inf: 5.686864400775745 , step_size: 1.0 , using inv_hess0: False
        Iter: 7 , cost: 0.023420212085108402 , |g|_inf: 1.5704063229963603 , step_size: 1.0 , using inv_hess0: False
        Iter: 8 , cost: 0.022180253777025063 , |g|_inf: 0.17433645466701125 , step_size: 1.0 , using inv_hess0: False
        Iter: 9 , cost: 0.022151908056769233 , |g|_inf: 0.11831447213364044 , step_size: 1.0 , using inv_hess0: False
        Iter: 10 , cost: 0.022133010800577987 , |g|_inf: 0.11402668398416971 , step_size: 1.0 , using inv_hess0: False
        Iter: 11 , cost: 0.022056436870526372 , |g|_inf: 0.21700710669019296 , step_size: 1.0 , using inv_hess0: False
        Iter: 12 , cost: 0.021886291988417073 , |g|_inf: 0.4594384674152714 , step_size: 1.0 , using inv_hess0: False
        Iter: 13 , cost: 0.021419477729493135 , |g|_inf: 0.8681741839339854 , step_size: 1.0 , using inv_hess0: False
        Iter: 14 , cost: 0.02028561455954908 , |g|_inf: 1.463264156139919 , step_size: 1.0 , using inv_hess0: False
        Iter: 15 , cost: 0.017710838497109766 , |g|_inf: 2.2247567158236277 , step_size: 1.0 , using inv_hess0: False
        Iter: 16 , cost: 0.01329633053477537 , |g|_inf: 2.744706991756995 , step_size: 1.0 , using inv_hess0: False
        Iter: 17 , cost: 0.007709379962193361 , |g|_inf: 2.2656282021799203 , step_size: 0.35986933832901696 , using inv_hess0: False
        Iter: 18 , cost: 0.0054833100677158155 , |g|_inf: 2.1711999804740865 , step_size: 1.0 , using inv_hess0: False
        Iter: 19 , cost: 0.0003740486325942903 , |g|_inf: 0.36510436662563045 , step_size: 1.0 , using inv_hess0: False
        Iter: 20 , cost: 7.488746110463646e-05 , |g|_inf: 0.24329588889891557 , step_size: 1.0 , using inv_hess0: False
        Iter: 21 , cost: 1.830477855363404e-05 , |g|_inf: 0.12302056237760803 , step_size: 1.0 , using inv_hess0: False
        Iter: 22 , cost: 4.411567691498245e-06 , |g|_inf: 0.05863970038611927 , step_size: 1.0 , using inv_hess0: False
        Iter: 23 , cost: 2.202310701587146e-09 , |g|_inf: 0.0019280333566841972 , step_size: 1.0 , using inv_hess0: False
        Iter: 24 , cost: 7.274525404517185e-12 , |g|_inf: 9.430503164691668e-05 , step_size: 1.0 , using inv_hess0: False
        Iter: 25 , cost: 9.541897903167501e-15 , |g|_inf: 3.6120283847345e-06 , step_size: 1.0 , using inv_hess0: False
        LBFGS done.
            Termination reason: RTOL_ACHIEVED
            Iterations: 26
            Cost evaluations: 28
            Gradient evaluations: 53
            Final cost: 3.9063012510091e-18
            Final |g|_inf: 4.128863472988538e-08
        x= [1. 1. 1. 1. 1.]

        Using previous final inverse Hessian as new initial Hessian after 5 iterations
        Iter: 0 , cost: 848.22 , |g|_inf: 2085.4 , step_size: 0.0004496668508702465 , using inv_hess0: False
        Iter: 1 , cost: 41.34323740111179 , |g|_inf: 178.40189326409805 , step_size: 1.0 , using inv_hess0: False
        Iter: 2 , cost: 16.826849424945813 , |g|_inf: 105.5379767341664 , step_size: 1.0 , using inv_hess0: False
        Iter: 3 , cost: 2.7868248969070386 , |g|_inf: 51.01396443585729 , step_size: 1.0 , using inv_hess0: False
        Iter: 4 , cost: 1.0446811105748832 , |g|_inf: 21.4422281910189 , step_size: 1.0 , using inv_hess0: False
        Iter: 5 , cost: 0.11343722694575001 , |g|_inf: 10.271531675646154 , step_size: 1.0 , using inv_hess0: True
        Iter: 6 , cost: 0.027903578598861556 , |g|_inf: 5.033211989929168 , step_size: 1.0 , using inv_hess0: True
        Iter: 7 , cost: 0.007612495012282753 , |g|_inf: 1.8882487205461977 , step_size: 1.0 , using inv_hess0: True
        Iter: 8 , cost: 0.0020264075816008036 , |g|_inf: 0.8379239722392997 , step_size: 1.0 , using inv_hess0: True
        Iter: 9 , cost: 0.0007145427190684071 , |g|_inf: 0.6949123244055606 , step_size: 1.0 , using inv_hess0: True
        Iter: 10 , cost: 7.039729253501432e-05 , |g|_inf: 0.2777101575526239 , step_size: 1.0 , using inv_hess0: True
        Iter: 11 , cost: 1.9016824157216738e-05 , |g|_inf: 0.13022925174117678 , step_size: 1.0 , using inv_hess0: True
        Iter: 12 , cost: 3.3308347005081375e-08 , |g|_inf: 0.004710133337606962 , step_size: 1.0 , using inv_hess0: True
        Iter: 13 , cost: 1.2938550580040016e-10 , |g|_inf: 0.00026003960834704943 , step_size: 1.0 , using inv_hess0: True
        Iter: 14 , cost: 5.996089267977567e-14 , |g|_inf: 1.0548330665030862e-05 , step_size: 1.0 , using inv_hess0: True
        Iter: 15 , cost: 2.4010557108165267e-16 , |g|_inf: 3.747210574866713e-07 , step_size: 1.0 , using inv_hess0: True
        LBFGS done.
            Termination reason: RTOL_ACHIEVED
            Iterations: 16
            Cost evaluations: 17
            Gradient evaluations: 33
            Final cost: 1.5871272067049831e-18
            Final |g|_inf: 2.8119372165952003e-08
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
    '''
    num_cost_evals: int = 0
    def __cost_with_counter(x):
        nonlocal num_cost_evals
        num_cost_evals += 1
        return cost(x)

    num_grad_evals: int = 0
    def __grad_with_counter(x):
        nonlocal  num_grad_evals
        num_grad_evals += 1
        return grad(x)

    iter: int = 0
    if isinstance(inv_hess0, LbfgsInverseHessianApproximation) and iter >= num_initial_iter:
        inv_hess = inv_hess0
    elif iter >= num_initial_iter:
        inv_hess = LbfgsInverseHessianApproximation(max_vector_pairs_stored, deque(), deque(), inv_hess0)
    else:
        inv_hess = LbfgsInverseHessianApproximation(max_vector_pairs_stored, deque(), deque(), None)

    x: np.ndarray = x0
    f: float        = __cost_with_counter(x)
    g: np.ndarray   = __grad_with_counter(x)
    # gradnorm: float = np.linalg.norm(g)
    gradnorm: float = np.max(np.abs(g))

    cost_history: typ.List[float] = [f]
    gradnorm_history: typ.List[float] = [gradnorm]

    p: np.ndarray = inv_hess.matvec(-g)
    if inv_hess.inv_hess0 is None:
        # print('np.linalg.norm(g) / 2.0=', np.linalg.norm(g) / 2.0)
        old_old_fval = f + np.linalg.norm(g) / 2.0
        # old_old_fval = f + gradnorm / 2.0
    else:
        old_old_fval = None
    line_search_result = line_search(__cost_with_counter, __grad_with_counter, x, p, g, f, old_old_fval)
    step_size: float = line_search_result[0]

    step_size_history: typ.List[float] = [step_size]

    def __display_iter_info():
        if display:
            print('Iter:', iter, ', cost:', f, ', |g|_inf:', gradnorm, ', step_size:', step_size,
                  ', using inv_hess0:', inv_hess.inv_hess0 is not None)

    __display_iter_info()
    f0: float = f
    gradnorm0: float = gradnorm

    termination_reason: LbfgsTerminationReason = LbfgsTerminationReason.MAXITER_REACHED
    while iter < max_iter:
        iter += 1
        if iter == num_initial_iter:
            if isinstance(inv_hess0, LbfgsInverseHessianApproximation):
                inv_hess = inv_hess0
            else:
                inv_hess = LbfgsInverseHessianApproximation(max_vector_pairs_stored, deque(), deque(), inv_hess0)

        x_old = x
        f_old = f
        g_old = g

        x = x + step_size * p
        f = line_search_result[3]
        g = __grad_with_counter(x)
        # gradnorm: float = np.linalg.norm(g)
        gradnorm = np.max(np.abs(g))

        cost_history.append(f)
        gradnorm_history.append(gradnorm)

        if gradnorm <= rtol * gradnorm0:
            termination_reason: LbfgsTerminationReason = LbfgsTerminationReason.RTOL_ACHIEVED
            break

        if (f_old - f) < stag_tol * f_old:
            termination_reason: LbfgsTerminationReason = LbfgsTerminationReason.DESCENT_STAGNATED
            break

        inv_hess.add_new_s_y_pair(x - x_old, g - g_old)
        p = inv_hess.matvec(-g)
        line_search_result = line_search(__cost_with_counter, __grad_with_counter, x, p, g, f)
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
        print('    Final |g|_inf:', gradnorm)

    return LbfgsResult(x, f, g, inv_hess,
                       iter, num_cost_evals, num_grad_evals,
                       cost_history, gradnorm_history, step_size_history,
                       termination_reason)


class LbfgsTerminationReason(Enum):
    MAXITER_REACHED = 0
    RTOL_ACHIEVED = 1
    DESCENT_STAGNATED = 2


class LbfgsResult(typ.NamedTuple):
    x: np.ndarray       # solution
    cost: float         # cost(x)
    grad: np.ndarray    # grad f(x)
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
    ss: typ.Deque[np.ndarray] # GETS MODIFIED! ss=[s_(k-1), s_(k-2), ..., s_(k-m)], s_i = x_(i+1) - x_i. Eq 7.18, left, on page 177
    yy: typ.Deque[np.ndarray] # GETS MODIFIED! yy=[y_(k-1), y_(k-2), ..., y_(k-m)], y_i = grad f_(i+1) - grad f_k. Eq 7.18, right, on page 177
    inv_hess0: typ.Callable[[np.ndarray], np.ndarray] = None # Initial inverse Hessian approximation

    def __post_init__(me) -> None:
        assert(me.m >= 0)
        assert(len(me.ss) == len(me.yy))
        while len(me.ss) > me.m:
            me.ss.pop()
        while len(me.yy) > me.m:
            me.yy.pop()

        if me.ss:
            x_shape = me.ss[0].shape
            for s in me.ss:
                assert(s.shape == x_shape)
            for y in me.yy:
                assert(y.shape == x_shape)

    def add_new_s_y_pair(
            me,
            s: np.ndarray,
            y: np.ndarray
    ) -> None:
        x_shape = s.shape
        assert(y.shape == x_shape)
        if me.ss:
            assert(me.ss[0].shape == x_shape)

        me.ss.appendleft(s)
        if len(me.ss) > me.m:
            me.ss.pop()

        me.yy.appendleft(y)
        if len(me.yy) > me.m:
            me.yy.pop()

    def apply_inv_hess0_k(me, x: np.ndarray) -> np.ndarray:
        if me.inv_hess0 is not None:
            return me.inv_hess0(x)
        else:
            if me.ss:
                gamma_k = np.sum(me.ss[0] * me.yy[0]) / np.sum(me.yy[0] * me.yy[0]) # <s_(k-1), y_(k-1)> / <y_(k-1), y_(k-1)>
            else:
                gamma_k = 1.0
            return gamma_k * x # H0_k = gamma_k*I

    def matvec(me, q: np.ndarray) -> np.ndarray:
        '''Computes
            r = H_k grad f_k
        via L-BFGS two-loop recursion.
        Algorithm 7.4 on page 178 of Nocedal and Wright.
        '''
        rhos = [1.0 / np.sum(y * s) for s, y in zip(me.ss, me.yy)] # equation 7.17 (left) on page 177
        alphas = []
        for s, y, rho in zip(me.ss, me.yy, rhos):
            alpha = rho * np.sum(s * q)
            q = q - alpha * y
            alphas.append(alpha)
        r = me.apply_inv_hess0_k(q)
        for s, y, rho, alpha in zip(reversed(me.ss), reversed(me.yy), reversed(rhos), reversed(alphas)):
            beta = rho * np.sum(y * r)
            r = r + s * (alpha - beta)
        return r

