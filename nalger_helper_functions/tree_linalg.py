import typing as typ
import numpy as np

# Elementwise operations for objects arranged in trees

__all__ = [
    'tree_binary_operation',
    'tree_elementwise_operation',
    'tree_reduce',
    'add',
    'sub',
    'mult',
    'scale',
    'div',
    'power',
    'abs',
    'leaf_sum',
    'leaf_dot',
    'leaf_norm',
    'sum',
    'dot',
    'normsquared',
    'norm',
    'all',
    'any',
    'eq',
    'lt',
    'le',
    'gt',
    'ge',
    'eq_scalar',
    'lt_scalar',
    'le_scalar',
    'gt_scalar',
    'ge_scalar',
    'elementwise_inverse',
]

_Tree = typ.TypeVar('_Tree')

def tree_binary_operation(
        op: typ.Callable[[typ.Any, typ.Any], typ.Any],
        U: _Tree,
        V: _Tree,
) -> _Tree:
    if isinstance(U, list):
        assert(len(U) == len(V))
        return [tree_binary_operation(op, u, v) for u, v in zip(U, V)]
    elif isinstance(U, tuple):
        return tuple(tree_binary_operation(op, list(U), list(V)))
    elif isinstance(U, dict):
        assert(isinstance(V, dict))
        assert(U.keys() == V.keys())
        W = dict()
        for k in U.keys():
            W[k] = tree_binary_operation(op, U[k], V[k])
        return W
    else:
        return op(U, V)


def tree_elementwise_operation(
        op: typ.Callable[[typ.Any], typ.Any],
        U: _Tree,
) -> _Tree:
    return tree_binary_operation(lambda u, v: op(u), U, U)


def tree_reduce(
        elementwise_op: typ.Callable[[typ.Any], typ.Any],
        combine_op: typ.Callable[[typ.Any, typ.Any], typ.Any],
        U: _Tree,
) -> typ.Any:
    def _combine(u, v):
        if u is None:
            return v
        else:
            return combine_op(u, v)

    s = [None]
    def _reduce(u):
        s[0] = _combine(s[0], elementwise_op(u))
        return None

    tree_elementwise_operation(_reduce, U)
    return s[0]


add    = lambda U, V:  tree_binary_operation(lambda u, v: u + v, U, V)
sub    = lambda U, V:  tree_binary_operation(lambda u, v: u - v, U, V)
mult   = lambda U, V:  tree_binary_operation(lambda u, v: u * v, U, V)
scale  = lambda U, c:  tree_binary_operation(lambda u, v: c * u, U, U)
div    = lambda U, V:  tree_binary_operation(lambda u, v: u / v, U, V)

power                  = lambda U, p:  tree_elementwise_operation(lambda u: u ** p, U)
abs                    = lambda U:     tree_elementwise_operation(lambda u: np.abs(u), U)
sqrt                   = lambda U:     tree_elementwise_operation(lambda u: np.sqrt(u), U)
elementwise_inverse    = lambda U:     tree_elementwise_operation(lambda u: 1.0 / u, U)
leaf_sum             = lambda U:     tree_elementwise_operation(lambda u: np.sum(u), U) # sums within leaves only

leaf_dot             = lambda U, V:  tree_binary_operation(lambda u, v: np.sum(u * v), U, V) # sums within leaves only
leaf_normsquared     = lambda U:     leaf_dot(U, U)
leaf_norm            = lambda U:     tree_elementwise_operation(lambda u: np.sqrt(u), leaf_normsquared(U))

ones = lambda U: tree_elementwise_operation(lambda u: np.ones(np.array(u).shape), U)
randn = lambda U: tree_elementwise_operation(lambda u: np.random.randn(*np.array(u).shape), U)

sum            = lambda U:     tree_reduce(np.sum, lambda u, v: u + v, U)
dot            = lambda U, V:  sum(mult(U, V))
normsquared    = lambda U:     dot(U, U)
norm           = lambda U:     np.sqrt(normsquared(U))

all = lambda U: tree_reduce(np.all, lambda u, v: np.logical_and(u, v), U)
any = lambda U: tree_reduce(np.any, lambda u, v: np.logical_or(u, v), U)

eq = lambda U, V: tree_binary_operation(lambda u, v: u == v, U, V)
lt = lambda U, V: tree_binary_operation(lambda u, v: u < v, U, V)
le = lambda U, V: tree_binary_operation(lambda u, v: u <= v, U, V)
gt = lambda U, V: tree_binary_operation(lambda u, v: u > v, U, V)
ge = lambda U, V: tree_binary_operation(lambda u, v: u >= v, U, V)

eq_scalar = lambda U, c: tree_elementwise_operation(lambda u: u == c, U)
lt_scalar = lambda U, c: tree_elementwise_operation(lambda u: u < c, U)
le_scalar = lambda U, c: tree_elementwise_operation(lambda u: u <= c, U)
gt_scalar = lambda U, c: tree_elementwise_operation(lambda u: u > c, U)
ge_scalar = lambda U, c: tree_elementwise_operation(lambda u: u >= c, U)

