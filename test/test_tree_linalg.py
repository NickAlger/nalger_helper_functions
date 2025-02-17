import numpy as np
from nalger_helper_functions.tree_linalg import *

def make_tree(x1, x2, x3, x4, x5):
    return [x1, {'a': x2, 'b': (x3, x4)}, x5]

def flatten(X):
    return np.concatenate([
        np.array(X[0]).reshape(-1),
        np.array(X[1]['a']).reshape(-1),
        np.array(X[1]['b'][0]).reshape(-1),
        np.array(X[1]['b'][1]).reshape(-1),
        np.array(X[2]).reshape(-1)
     ])

def check_if_tree_is_structured_correctly(X):
    assert(isinstance(X, list))
    assert(len(X) == 3)
    A, B, C = tuple(X)
    assert(np.array(A).shape == ())
    assert(isinstance(B, dict))
    assert(B.keys() == {'a': None, 'b': None}.keys())
    Ba = B['a']
    Bb = B['b']
    assert(isinstance(Ba, np.ndarray))
    assert(B['a'].shape == (2,3))
    assert(isinstance(Bb, tuple))
    assert(len(Bb) == 2)
    Bb1, Bb2 = B['b']
    assert(np.array(Bb1).shape == ())
    assert(isinstance(Bb2, np.ndarray))
    assert(Bb2.shape == (4,))
    assert(np.array(C).shape == ())

# add()

t1 = int(np.random.randint(1,10))
t2 = np.random.randn(2,3)
t3 = float(np.random.randn())
t4 = np.random.randn(4)
t5 = np.random.randn(1).reshape(())

T = make_tree(t1, t2, t3, t4, t5)
check_if_tree_is_structured_correctly(T)

s1 = int(np.random.randint(10))
s2 = np.random.randn(2,3)
s3 = float(np.random.randn())
s4 = np.random.randn(4)
s5 = np.random.randn(1).reshape(())

S = make_tree(s1, s2, s3, s4, s5)
check_if_tree_is_structured_correctly(S)

print('T:')
print(T)

print('S:')
print(S)

T_plus_S = add(T, S)
check_if_tree_is_structured_correctly(T_plus_S)

print('T_plus_S:')
print(T_plus_S)

err_add = np.linalg.norm(flatten(T_plus_S) - (flatten(T) + flatten(S)))
print('err_add=', err_add)

# sub()

T_minus_S = sub(T, S)
check_if_tree_is_structured_correctly(T_minus_S)
err_sub = np.linalg.norm(flatten(T_minus_S) - (flatten(T) - flatten(S)))
print('err_sub=', err_sub)

# mult()

T_times_S = mult(T, S)
check_if_tree_is_structured_correctly(T_times_S)
err_mult = np.linalg.norm(flatten(T_times_S) - (flatten(T) * flatten(S)))
print('err_mult=', err_mult)

# scale()

c = np.random.randn()
T_times_c = scale(T, c)
check_if_tree_is_structured_correctly(T_times_c)
err_scale = np.linalg.norm(flatten(T_times_c) - (c * flatten(T)))
print('err_scale=', err_scale)

# power()

p = 3
T_to_the_p = power(T, p)
check_if_tree_is_structured_correctly(T_to_the_p)
err_power = np.linalg.norm(flatten(T_to_the_p) - flatten(T)**p)
print('err_power=', err_power)

# abs()

abs_T = abs(T)
check_if_tree_is_structured_correctly(abs_T)
err_abs = np.linalg.norm(flatten(abs_T) - np.abs(flatten(T)))
print('err_abs=', err_abs)

# elementwise_inverse()

iT_true = make_tree(1.0/t1, 1.0/t2, 1.0/t3, 1.0/t4, 1.0/t5)
iT = elementwise_inverse(T)
err_elementwise_inverse = np.linalg.norm(flatten(iT_true) - flatten(iT))
print('err_elementwise_inverse=', err_elementwise_inverse)

# sum_leaves()

TSL = leaf_sum(T)
TSL_true = make_tree(np.sum(t1), np.sum(t2), np.sum(t3), np.sum(t4), np.sum(t5))

print('TSL=', TSL)
print('TSL_true=', TSL_true)

err_sum_leaves = norm(abs(sub(TSL, TSL_true)))
print('err_sum_leaves=', err_sum_leaves)

# dot_leaves()

T_dot_S_leaves = leaf_dot(T, S)
T_dot_S_leaves_true = make_tree(np.sum(t1*s1), np.sum(t2*s2), np.sum(t3*s3), np.sum(t4*s4), np.sum(t5*s5))

print('T_dot_S_leaves=', T_dot_S_leaves)
print('T_dot_S_leaves_true=', T_dot_S_leaves_true)

err_dot_leaves = norm(abs(sub(T_dot_S_leaves, T_dot_S_leaves_true)))
print('err_dot_leaves=', err_dot_leaves)


# sum()

sum_T = sum(T)
err_sum = np.abs(sum_T - np.sum(flatten(T)))
print('err_sum=', err_sum)

# dot()

T_dot_S = dot(T, S)
err_dot = np.abs(T_dot_S - np.dot(flatten(T), flatten(S)))
print('err_dot=', err_dot)

# normsquared()

normsq_T = normsquared(T)
err_normsquared = np.abs(normsq_T - np.linalg.norm(flatten(T))**2)
print('err_normsquared=', err_normsquared)

# norm()

norm_T = norm(T)
err_norm = np.abs(norm_T - np.linalg.norm(flatten(T)))
print('err_norm=', err_norm)

# all()

t1_true = True
t2_true = np.array([[True, True, True], [True, True, True]])
t3_true = True
t4_true = np.array([True, True, True, True])
t5_true = np.array(True)

T_true = make_tree(t1_true, t2_true, t3_true, t4_true, t5_true)
check_if_tree_is_structured_correctly(T_true)

t1_false = False
t2_false = np.array([[False, False, False], [False, False, False]])
t3_false = False
t4_false = np.array([False, False, False, False])
t5_false = np.array(False)

T_false = make_tree(t1_false, t2_false, t3_false, t4_false, t5_false)
check_if_tree_is_structured_correctly(T_false)

t1_mixed = False
t2_mixed = np.array([[False, True, False], [False, False, False]])
t3_mixed = False
t4_mixed = np.array([False, False, False, False])
t5_mixed = np.array(True)

T_mixed = make_tree(t1_mixed, t2_mixed, t3_mixed, t4_mixed, t5_mixed)
check_if_tree_is_structured_correctly(T_mixed)

all_test1_passed = (all(T_true) == True)
all_test2_passed = (all(T_mixed) == False)
all_test3_passed = (all(T_false) == False)
print('all_test1_passed=', all_test1_passed)
print('all_test2_passed=', all_test2_passed)
print('all_test3_passed=', all_test3_passed)

# any()

any_test1_passed = (all(T_true) == True)
any_test2_passed = (any(T_mixed) == True)
any_test3_passed = (any(T_false) == False)
print('any_test1_passed=', any_test1_passed)
print('any_test2_passed=', any_test2_passed)
print('any_test3_passed=', any_test3_passed)

# eq()

T_equals_S = eq(T, S)
check_if_tree_is_structured_correctly(T_equals_S)
eq_test1_passed = np.all(flatten(T_equals_S) == (flatten(T) == flatten(S)))
print('eq_test1_passed=', eq_test1_passed)

T_equals_T = eq(T, T)
check_if_tree_is_structured_correctly(T_equals_T)
eq_test2_passed = np.all(flatten(T_equals_T) == (flatten(T) == flatten(T)))
print('eq_test2_passed=', eq_test2_passed)

t2b = t2.copy()
t2b[0,1] += 0.5
t4b = t4 - 1.8

Tb = make_tree(t1, t2b, t3, t4b, t5)
check_if_tree_is_structured_correctly(Tb)

T_equals_Tb = eq(T, Tb)
check_if_tree_is_structured_correctly(T_equals_Tb)
eq_test3_passed = np.all(flatten(T_equals_Tb) == (flatten(T) == flatten(Tb)))
print('eq_test3_passed=', eq_test3_passed)

# lt()

T_lt_S = lt(T, S)
check_if_tree_is_structured_correctly(T_lt_S)
lt_test1_passed = np.all(flatten(T_lt_S) == (flatten(T) < flatten(S)))
print('lt_test1_passed=', lt_test1_passed)

T_lt_T = lt(T, T)
check_if_tree_is_structured_correctly(T_lt_T)
lt_test2_passed = np.all(flatten(T_lt_T) == (flatten(T) < flatten(T)))
print('lt_test2_passed=', lt_test2_passed)

T_lt_Tb = lt(T, Tb)
check_if_tree_is_structured_correctly(T_lt_Tb)
lt_test3_passed = np.all(flatten(T_lt_Tb) == (flatten(T) < flatten(Tb)))
print('lt_test3_passed=', lt_test3_passed)

# le()

T_le_S = le(T, S)
check_if_tree_is_structured_correctly(T_le_S)
le_test1_passed = np.all(flatten(T_le_S) == (flatten(T) <= flatten(S)))
print('le_test1_passed=', le_test1_passed)

T_le_T = le(T, T)
check_if_tree_is_structured_correctly(T_le_T)
le_test2_passed = np.all(flatten(T_le_T) == (flatten(T) <= flatten(T)))
print('le_test2_passed=', le_test2_passed)

T_le_Tb = le(T, Tb)
check_if_tree_is_structured_correctly(T_le_Tb)
le_test3_passed = np.all(flatten(T_le_Tb) == (flatten(T) <= flatten(Tb)))
print('le_test3_passed=', le_test3_passed)

# gt()

T_gt_S = gt(T, S)
check_if_tree_is_structured_correctly(T_gt_S)
gt_test1_passed = np.all(flatten(T_gt_S) == (flatten(T) > flatten(S)))
print('gt_test1_passed=', gt_test1_passed)

T_gt_T = gt(T, T)
check_if_tree_is_structured_correctly(T_gt_T)
gt_test2_passed = np.all(flatten(T_gt_T) == (flatten(T) > flatten(T)))
print('gt_test2_passed=', gt_test2_passed)

T_gt_Tb = gt(T, Tb)
check_if_tree_is_structured_correctly(T_gt_Tb)
gt_test3_passed = np.all(flatten(T_gt_Tb) == (flatten(T) > flatten(Tb)))
print('gt_test3_passed=', gt_test3_passed)

# ge()

T_ge_S = ge(T, S)
check_if_tree_is_structured_correctly(T_ge_S)
ge_test1_passed = np.all(flatten(T_ge_S) == (flatten(T) >= flatten(S)))
print('ge_test1_passed=', ge_test1_passed)

T_ge_T = ge(T, T)
check_if_tree_is_structured_correctly(T_ge_T)
ge_test2_passed = np.all(flatten(T_ge_T) == (flatten(T) >= flatten(T)))
print('ge_test2_passed=', ge_test2_passed)

T_ge_Tb = ge(T, Tb)
check_if_tree_is_structured_correctly(T_ge_Tb)
ge_test3_passed = np.all(flatten(T_ge_Tb) == (flatten(T) >= flatten(Tb)))
print('ge_test3_passed=', ge_test3_passed)

# eq_scalar()

c = t2[1,1]
T_eq_c = eq_scalar(T, c)
check_if_tree_is_structured_correctly(T_eq_c)
eq_scalar_test1_passed = np.all(flatten(T_eq_c) == (flatten(T) == c))
print('eq_scalar_test1_passed=', eq_scalar_test1_passed)

c = t1
T_eq_c = eq_scalar(T, c)
check_if_tree_is_structured_correctly(T_eq_c)
eq_scalar_test2_passed = np.all(flatten(T_eq_c) == (flatten(T) == c))
print('eq_scalar_test2_passed=', eq_scalar_test2_passed)

# lt_scalar()

c = t2[1,1]
T_lt_c = lt_scalar(T, c)
check_if_tree_is_structured_correctly(T_lt_c)
lt_scalar_test1_passed = np.all(flatten(T_lt_c) == (flatten(T) < c))
print('lt_scalar_test1_passed=', lt_scalar_test1_passed)

c = t1
T_lt_c = lt_scalar(T, c)
check_if_tree_is_structured_correctly(T_lt_c)
lt_scalar_test2_passed = np.all(flatten(T_lt_c) == (flatten(T) < c))
print('lt_scalar_test2_passed=', lt_scalar_test2_passed)

# le_scalar()

c = t2[1,1]
T_le_c = le_scalar(T, c)
check_if_tree_is_structured_correctly(T_le_c)
le_scalar_test1_passed = np.all(flatten(T_le_c) == (flatten(T) <= c))
print('le_scalar_test1_passed=', le_scalar_test1_passed)

c = t1
T_le_c = le_scalar(T, c)
check_if_tree_is_structured_correctly(T_le_c)
le_scalar_test2_passed = np.all(flatten(T_le_c) == (flatten(T) <= c))
print('le_scalar_test2_passed=', le_scalar_test2_passed)

# gt_scalar()

c = t2[1,1]
T_gt_c = gt_scalar(T, c)
check_if_tree_is_structured_correctly(T_gt_c)
gt_scalar_test1_passed = np.all(flatten(T_gt_c) == (flatten(T) > c))
print('gt_scalar_test1_passed=', gt_scalar_test1_passed)

c = t1
T_gt_c = gt_scalar(T, c)
check_if_tree_is_structured_correctly(T_gt_c)
gt_scalar_test2_passed = np.all(flatten(T_gt_c) == (flatten(T) > c))
print('gt_scalar_test2_passed=', gt_scalar_test2_passed)

# ge_scalar()

c = t2[1,1]
T_ge_c = ge_scalar(T, c)
check_if_tree_is_structured_correctly(T_ge_c)
ge_scalar_test1_passed = np.all(flatten(T_ge_c) == (flatten(T) >= c))
print('ge_scalar_test1_passed=', ge_scalar_test1_passed)

c = t1
T_ge_c = ge_scalar(T, c)
check_if_tree_is_structured_correctly(T_ge_c)
ge_scalar_test2_passed = np.all(flatten(T_ge_c) == (flatten(T) >= c))
print('ge_scalar_test2_passed=', ge_scalar_test2_passed)
