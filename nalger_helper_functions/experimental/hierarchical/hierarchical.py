import numpy as np
import numpy.typing as npt
import typing as typ
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm
from functools import cached_property
from dataclasses import dataclass
import collections
import itertools
import functools as ft

import jax.numpy as jnp
import jax


# Want: Permutation is a 1D array of ints
Permutation = typ.Sequence[int]  # shape=(num_points_total,)


def check_permutation(P: Permutation):
    assert (np.all(np.sort(P) == np.arange(len(P))))


def invert_permutation(P: Permutation) -> Permutation:
    inv_P = np.argsort(P)
    assert (np.all(inv_P[P] == np.arange(len(P))))
    assert (np.all(P[inv_P] == np.arange(len(P))))
    return tuple(list(inv_P))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ContiguousIndexSet:
    start:  int
    stop:   int

    def __post_init__(me):
        assert (me.start >= 0)
        assert (me.stop >= me.start)

    @cached_property
    def size(me) -> int:
        return me.stop - me.start

    @cached_property
    def slice(me) -> slice:
        return slice(me.start, me.stop)

    def indices(me) -> npt.ArrayLike:
        return np.arange(me.start, me.stop, dtype=int)

    def __contains__(me, x: int):
        return (me.start <= x) and (x < me.stop)

    @ft.cached_property
    def data(me) -> typ.Tuple[int, int]:
        return me.start, me.stop

    def tree_flatten(me):
        return (me.data, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ClusterTree:
    children:   typ.List['ClusterTree']  # OK
    inds:       ContiguousIndexSet

    def __post_init__(me):
        if me.children:
            starts = jnp.array(sorted([c.inds.start for c in me.children]))
            stops = jnp.array(sorted([c.inds.stop for c in me.children]))
            assert (jnp.all(stops[:-1] == starts[1:]))  # child index sets partition an interval
            assert (starts[0] == me.inds.start)
            assert (stops[-1] == me.inds.stop)

    @cached_property
    def size(me) -> int:
        return me.inds.size

    @cached_property
    def is_leaf(me) -> bool:
        return len(me.children) == 0

    @ft.cached_property
    def data(me) -> typ.Tuple[typ.List['ClusterTree'], ContiguousIndexSet]:
        return me.children, me.inds

    def tree_flatten(me):
        return (me.data, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def index_set_subset(
        A: ContiguousIndexSet,  # proposed subset
        B: ContiguousIndexSet  # proposed superset
) -> bool:
    '''True iff A is a subset of B'''
    return (B.start <= A.start) and (A.stop <= B.stop)


def index_sets_intersect(
        A: ContiguousIndexSet,
        B: ContiguousIndexSet
) -> bool:
    '''True iff A and B intersect'''
    return (A.start < B.stop) and (B.start < A.stop)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Box:
    min_pt: jnp.ndarray
    max_pt: jnp.ndarray

    def __post_init__(me):
        assert (len(me.min_pt.shape) == 1)
        assert (len(me.max_pt.shape) == 1)
        assert (jnp.all(me.max_pt >= me.min_pt))

    @cached_property
    def d(me) -> int:
        return len(me.min_pt.shape)

    @cached_property
    def widths(me) -> npt.ArrayLike:
        return me.max_pt - me.min_pt

    @cached_property
    def widest_dimension(me) -> int:
        return jnp.argmax(me.widths)

    @cached_property
    def narrowest_dimension(me) -> int:
        return jnp.argmin(me.widths)

    @cached_property
    def diameter(me) -> float:
        return jnp.linalg.norm(me.max_pt - me.min_pt)

    @ft.cached_property
    def data(me) -> typ.Tuple[jnp.ndarray, jnp.ndarray]:
        return me.min_pt, me.max_pt

    def tree_flatten(me):
        return (me.data, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def boxes_distance(A: Box, B: Box):
    delta1 = A.min_pt - B.max_pt
    delta2 = B.min_pt - A.max_pt
    u = np.max(np.array([np.zeros(len(delta1)), delta1]), axis=0)
    v = np.max(np.array([np.zeros(len(delta2)), delta2]), axis=0)
    dist = np.linalg.norm(np.concatenate([u, v]))
    return dist


def cluster_pointcloud(
        cluster_inds:   ContiguousIndexSet,  # "sigma", "tau"
        pointcloud:     jnp.ndarray,  # shape=(num_points_total, spatial_dimension)
        perm_e2i:       Permutation,  # len=(num_points_total)
) -> jnp.ndarray:
    _, d = pointcloud.shape
    return pointcloud[perm_e2i[cluster_inds.slice], :].reshape((-1, d))  # reshape in case there are one or zero points


def pointcloud_bounding_box(
        pointcloud: jnp.ndarray,  # shape=(num_points, spatial_dimension)
) -> Box:
    min_pt = jnp.min(pointcloud, axis=0).reshape(-1)
    max_pt = jnp.max(pointcloud, axis=0).reshape(-1)
    return Box(min_pt, max_pt)


def subdivide_cluster_index_set(
        cluster_inds:   ContiguousIndexSet,
        pointcloud:     jnp.ndarray,  # shape=(num_points_total, spatial_dimension)
        perm_e2i:       Permutation,  # len=(num_points_total) GETS MODIFIED!
) -> typ.Tuple[ContiguousIndexSet, ContiguousIndexSet]:  # (left, right)
    n = cluster_inds.size
    N, d = pointcloud.shape

    pp = cluster_pointcloud(cluster_inds, pointcloud, perm_e2i)
    B = pointcloud_bounding_box(pp)
    split_direction = B.widest_dimension
    sort_inds = jnp.argsort(pp[:, split_direction].reshape(-1))

    left_sort = sort_inds[: int(n / 2)]
    right_sort = sort_inds[int(n / 2):]

    left_start = cluster_inds.start
    left_stop = cluster_inds.start + len(left_sort)
    right_start = left_stop
    right_stop = cluster_inds.stop

    left_inds = ContiguousIndexSet(left_start, left_stop)
    right_inds = ContiguousIndexSet(right_start, right_stop)

    left_perm = perm_e2i[cluster_inds.slice][left_sort].copy()
    right_perm = perm_e2i[cluster_inds.slice][right_sort].copy()

    perm_e2i[left_start: left_stop] = left_perm
    perm_e2i[right_start: right_stop] = right_perm
    return left_inds, right_inds


def build_cluster_tree_helper(
        inds:               ContiguousIndexSet,
        pointcloud:         jnp.ndarray,
        perm_e2i:           Permutation,
        min_cluster_size:   int
) -> ClusterTree:
    if inds.size > min_cluster_size:
        inds0, inds1 = subdivide_cluster_index_set(inds, pointcloud, perm_e2i)
        child0 = build_cluster_tree_helper(inds0, pointcloud,
                                           perm_e2i, min_cluster_size)
        child1 = build_cluster_tree_helper(inds1, pointcloud,
                                           perm_e2i, min_cluster_size)
        children = [child0, child1]
        return ClusterTree([child0, child1], inds)
    else:
        children = []

    return ClusterTree(children, inds)


def build_cluster_tree_geometric(
        pointcloud:         jnp.ndarray,  # shape=(num_pts_N, geometric_dimension_d)
        min_cluster_size:   int = 16,  # do not subdivide cluster if it is smaller than this
) -> typ.Tuple[Permutation, Permutation, ClusterTree]:  # e2i, i2e, ct
    assert (min_cluster_size > 1)
    N, d = pointcloud.shape
    root_inds = ContiguousIndexSet(0, N)
    perm_e2i = np.arange(N)  # gets modified
    root = build_cluster_tree_helper(root_inds,
                                     pointcloud,
                                     perm_e2i,
                                     min_cluster_size)

    perm_i2e = invert_permutation(perm_e2i)
    check_permutation(perm_e2i)
    check_permutation(perm_i2e)
    return perm_e2i, perm_i2e, root


def visualize_cluster_tree_2d(
        perm_e2i:   Permutation,
        ct:         ClusterTree,
        pointcloud: jnp.ndarray,
        fig_max_size=8
):
    N, d = pointcloud.shape
    assert (d == 2)

    B = pointcloud_bounding_box(pointcloud)

    max_levels = 0
    working_nodes = [(0, ct)]
    while working_nodes:
        level, node = working_nodes.pop()
        working_nodes += [(level + 1, c) for c in node.children]
        max_levels = np.max([max_levels, level + 1])

    plt.figure(figsize=(fig_max_size, fig_max_size * B.widths[1] / B.widths[0]))
    color = cm.viridis(np.linspace(0, 1, max_levels))
    working_nodes = [(0, ct)]
    while working_nodes:
        level, node = working_nodes.pop()
        working_nodes += [(level + 1, c) for c in node.children]
        node_pp = cluster_pointcloud(node.inds, pointcloud, perm_e2i)
        B = pointcloud_bounding_box(node_pp)

        c = color[level]
        rect = patches.Rectangle(tuple(B.min_pt), B.widths[0], B.widths[1],
                                 linewidth=1, edgecolor=None, facecolor=c)
        plt.gca().add_patch(rect)

    perm_i2e = invert_permutation(perm_e2i)
    ax = plt.gca()
    plt.scatter(pointcloud[:, 0], pointcloud[:, 1], s=2, c='k')
    for ii, txt in enumerate(list(perm_i2e)):
        ax.annotate(txt, (pointcloud[ii, 0], pointcloud[ii, 1]))
    plt.axis('equal')
    plt.show()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BlockClusterTree:
    children:       typ.List['BlockClusterTree']
    row_ct:         'ClusterTree'
    col_ct:         'ClusterTree'
    is_admissible:  bool

    def __post_init__(me):
        if me.is_admissible:
            assert (me.is_leaf)

        required_blocks = []
        if me.is_admissible:
            pass

        elif me.row_ct.is_leaf and me.col_ct.is_leaf:
            pass

        elif me.row_ct.is_leaf:
            for c in me.col_ct.children:
                required_blocks.append((me.row_ct.inds, c.inds))

        elif me.col_ct.is_leaf:
            for r in me.row_ct.children:
                required_blocks.append((r.inds, me.col_ct.inds))

        else:
            for r in me.row_ct.children:
                for c in me.col_ct.children:
                    required_blocks.append((r.inds, c.inds))

        actual_blocks = []
        for x in me.children:
            actual_blocks.append((x.row_inds, x.col_inds))

        assert (collections.Counter(actual_blocks) ==
                collections.Counter(required_blocks))

    @cached_property
    def row_inds(me) -> ContiguousIndexSet:
        return me.row_ct.inds

    @cached_property
    def col_inds(me) -> ContiguousIndexSet:
        return me.col_ct.inds

    @cached_property
    def shape(me) -> typ.Tuple[int, int]:
        return (me.row_inds.size, me.col_inds.size)

    @cached_property
    def size(me) -> int:
        return np.prod(me.shape)

    @cached_property  # not cached: might get refined later?
    def is_leaf(me) -> bool:
        return len(me.children) == 0

    @ft.cached_property
    def num_blocks(me) -> int:
        if me.is_leaf:
            return 1
        else:
            return sum([c.num_blocks for c in me.children])

    @cached_property
    def slices(me) -> typ.Tuple[slice, slice]:
        return me.row_inds.slice, me.col_inds.slice

    @ft.cached_property
    def data(me) -> typ.Tuple[typ.List['BlockClusterTree'], ClusterTree, ClusterTree, bool]:
        return me.children, me.row_ct, me.col_ct, me.is_admissible

    def tree_flatten(me):
        return (me.data, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def standard_admissibility(row_inds:        ContiguousIndexSet,
                           col_inds:        ContiguousIndexSet,
                           row_pointcloud:  jnp.ndarray,
                           col_pointcloud:  jnp.ndarray,
                           row_perm_e2i:    Permutation,
                           col_perm_e2i:    Permutation,
                           eta:             float,
) -> bool:
    pp_row = cluster_pointcloud(row_inds, row_pointcloud, row_perm_e2i)
    pp_col = cluster_pointcloud(col_inds, col_pointcloud, col_perm_e2i)
    R = pointcloud_bounding_box(pp_row)
    C = pointcloud_bounding_box(pp_col)
    return boxes_distance(R, C) >= eta * np.min([R.diameter, C.diameter])


def build_block_cluster_tree(
        row_ct:         ClusterTree,
        col_ct:         ClusterTree,
        row_pointcloud: jnp.ndarray,
        col_pointcloud: jnp.ndarray,
        row_perm_e2i:   Permutation,
        col_perm_e2i:   Permutation,
        admissibility_eta: float = 1.0
) -> BlockClusterTree:
    assert (admissibility_eta > 0.0)
    assert (len(row_perm_e2i) == row_pointcloud.shape[0])
    assert (len(col_perm_e2i) == col_pointcloud.shape[0])

    is_admissible = standard_admissibility(row_ct.inds, col_ct.inds,
                                           row_pointcloud, col_pointcloud,
                                           row_perm_e2i, col_perm_e2i,
                                           admissibility_eta)
    children = []
    if is_admissible:
        pass
    elif row_ct.is_leaf and col_ct.is_leaf:  # dense block
        pass
    elif row_ct.is_leaf:
        for c in col_ct.children:
            x = build_block_cluster_tree(row_ct, c,
                                         row_pointcloud,
                                         col_pointcloud,
                                         row_perm_e2i,
                                         col_perm_e2i,
                                         admissibility_eta)
            children.append(x)
    elif col_ct.is_leaf:
        for r in row_ct.children:
            x = build_block_cluster_tree(r, col_ct,
                                         row_pointcloud,
                                         col_pointcloud,
                                         row_perm_e2i,
                                         col_perm_e2i,
                                         admissibility_eta)
            children.append(x)
    else:
        for r in row_ct.children:
            for c in col_ct.children:
                x = build_block_cluster_tree(r, c,
                                             row_pointcloud,
                                             col_pointcloud,
                                             row_perm_e2i,
                                             col_perm_e2i,
                                             admissibility_eta)
                children.append(x)

    return BlockClusterTree(children, row_ct, col_ct, is_admissible)


def visualize_block_cluster_tree(bct: BlockClusterTree):
    A = np.zeros(bct.shape)
    working_nodes = [bct]
    while working_nodes:
        node = working_nodes.pop()
        working_nodes += node.children
        if node.is_admissible:
            A[node.slices] = np.random.rand()
        elif node.is_leaf:
            A[node.slices] = 2.0
    plt.matshow(A)


def polynomial_pointcloud_basis(
        pointcloud:         jnp.ndarray,  # shape=(N, d)
        polynomial_order_k: int  # 2x^2 + xyz + 3y + 1: k=3
) -> jnp.ndarray:  # shape=(N, d^num_polys)
    N, d = pointcloud.shape
    k = np.min([N - 1, polynomial_order_k])
    center = 0.5 * (np.max(pointcloud, axis=0) + np.min(pointcloud, axis=0))
    scale = 0.5 * (np.max(pointcloud, axis=0) - np.min(pointcloud, axis=0))
    xx = (pointcloud - center) / scale
    polys = list()
    for exponents in itertools.product(range(k + 1), repeat=d):
        if np.sum(exponents) <= k:
            ee = np.array(exponents).reshape((1, d))
            poly = np.prod(np.power(xx, ee), axis=1)
            polys.append(poly)
    B = np.array(polys).reshape((N, -1)).T
    Q, _, _ = np.linalg.svd(B, 0)
    return Q

