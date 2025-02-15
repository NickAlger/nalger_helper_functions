import typing as typ
import numpy as np

# Elementwise operations for objects arranged in trees

__all__ = [
    'tree_binary_operation',
    'tree_elementwise_operation',
    'tree_reduce',
    'tree_add',
    'tree_sub',
    'tree_mult',
    'tree_scale',
    'tree_power',
    'tree_abs',
    'tree_sum',
    'tree_dot',
    'tree_all',
    'tree_any',
    'tree_eq',
    'tree_lt',
    'tree_le',
    'tree_gt',
    'tree_ge',
    'tree_eq_scalar',
    'tree_lt_scalar',
    'tree_le_scalar',
    'tree_gt_scalar',
    'tree_ge_scalar',
]

_Tree = typ.Any

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


tree_add    = lambda U, V:  tree_binary_operation(lambda u, v: u + v, U, V)
tree_sub    = lambda U, V:  tree_binary_operation(lambda u, v: u - v, U, V)
tree_mult   = lambda U, V:  tree_binary_operation(lambda u, v: u * v, U, V)
tree_scale  = lambda U, c:  tree_binary_operation(lambda u, v: c * u, U, U)

tree_power  = lambda U, p:  tree_elementwise_operation(lambda u: u**p, U)
tree_abs    = lambda U:     tree_elementwise_operation(lambda u: np.abs(u), U)
tree_sum    = lambda U:     tree_reduce(np.sum, lambda u, v: u + v, U)
tree_dot    = lambda U, V:  tree_sum(tree_mult(U, V))

tree_all = lambda U: tree_reduce(np.all, lambda u, v: np.logical_and(u, v), U)
tree_any = lambda U: tree_reduce(np.any, lambda u, v: np.logical_or(u, v), U)

tree_eq = lambda U, V: tree_binary_operation(lambda u, v: u == v, U, V)
tree_lt = lambda U, V: tree_binary_operation(lambda u, v: u <  v, U, V)
tree_le = lambda U, V: tree_binary_operation(lambda u, v: u <= v, U, V)
tree_gt = lambda U, V: tree_binary_operation(lambda u, v: u >  v, U, V)
tree_ge = lambda U, V: tree_binary_operation(lambda u, v: u >= v, U, V)

tree_eq_scalar = lambda U, c: tree_elementwise_operation(lambda u: u == c, U)
tree_lt_scalar = lambda U, c: tree_elementwise_operation(lambda u: u <  c, U)
tree_le_scalar = lambda U, c: tree_elementwise_operation(lambda u: u <= c, U)
tree_gt_scalar = lambda U, c: tree_elementwise_operation(lambda u: u >  c, U)
tree_ge_scalar = lambda U, c: tree_elementwise_operation(lambda u: u >= c, U)



