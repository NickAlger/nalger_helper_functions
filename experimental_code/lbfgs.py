from __future__ import annotations
import numpy as np
import typing as typ
from collections import deque
from scipy.optimize import line_search
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
        inv_hess0: typ.Union[typ.Callable[[np.ndarray], np.ndarray], InverseHessianApproximation]=None,
) -> LbfgsResult:
    if isinstance(inv_hess0, InverseHessianApproximation):
        inv_hess = inv_hess0
    else:
        inv_hess = InverseHessianApproximation(max_vector_pairs_stored, deque(), deque(), inv_hess0)

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

    x: np.ndarray = x0
    f: float        = __cost_with_counter(x)
    g: np.ndarray   = __grad_with_counter(x)
    # gradnorm: float = np.linalg.norm(g)
    gradnorm: float = np.max(np.abs(g))

    cost_history: typ.List[float] = [f]
    gradnorm_history: typ.List[float] = [gradnorm]

    p: np.ndarray = inv_hess.solve_quasi_newton_system(-g)
    line_search_result = line_search(__cost_with_counter, __grad_with_counter, x, p, g, f)
    # line_search_result = line_search(cost, grad, x, p, g, f)
    # s = 3.5e-4
    # line_search_result = [s, None, None, cost(x + s*p)]
    step_size: float = line_search_result[0]

    step_size_history: typ.List[float] = [step_size]

    def __display_iter_info():
        if display:
            print('Iter:', iter, ', cost:', f, ', ||g||:', gradnorm, ', step_size:', step_size)

    __display_iter_info()
    f0: float = f
    gradnorm0: float = gradnorm

    termination_reason: LbfgsTerminationReason = LbfgsTerminationReason.MAXITER_REACHED
    while iter < max_iter:
        iter += 1
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
        p = inv_hess.solve_quasi_newton_system(-g)
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
        print('    Final ||g||:', gradnorm)

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
    inv_hess: InverseHessianApproximation   # L-BFGS approximation to the inverse Hessian at x
    iter: int
    num_cost_evals: int
    num_grad_evals: int
    cost_history: typ.List[float]      # [cost(x0), cost(x1), ..., cost(x)]
    gradnorm_history: typ.List[float]      # [grad(x0), grad(x1), ..., grad(x)]
    step_size_history: typ.List[float] # [a0, a1, ...], where x1 = x0 + a0*p, x2 = x1 + a1*p, ...
    termination_reason: LbfgsTerminationReason


@dataclass(frozen=True)
class InverseHessianApproximation:
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

    def solve_quasi_newton_system(me, grad_fk: np.ndarray) -> np.ndarray:
        '''Computes
            r = H_k grad f_k
        via L-BFGS two-loop recursion.
        Algorithm 7.4 on page 178 of Nocedal and Wright.
        '''
        rhos = [1.0 / np.sum(y * s) for s, y in zip(me.ss, me.yy)] # equation 7.17 (left) on page 177
        q = grad_fk
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


from scipy.optimize import minimize, rosen, rosen_der

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
result = lbfgs(rosen, rosen_der, x0, rtol=1e-10)

result = lbfgs(rosen, rosen_der, x0, inv_hess0=result.inv_hess, rtol=1e-10)

# result = lbfgs(rosen, rosen_der, np.array([1.3, 0.7, 0.8, 1.9, 1.2]), inv_hess0=lambda x: x)

# res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
#                options={'gtol': 1e-6, 'disp': True})

# res = minimize(rosen, x0, method='L-BFGS-B', jac=rosen_der, tol=1e-10, options={'gtol': 1e-15, 'disp': True})
res = minimize(rosen, x0, method='BFGS', jac=rosen_der, tol=1e-10, options={'gtol': 1e-15, 'disp': True})