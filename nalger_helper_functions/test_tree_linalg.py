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

# tree_add()

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

T_plus_S = tree_add(T, S)
check_if_tree_is_structured_correctly(T_plus_S)

print('T_plus_S:')
print(T_plus_S)

err_tree_add = np.linalg.norm(flatten(T_plus_S) - (flatten(T) + flatten(S)))
print('err_tree_add=', err_tree_add)

# tree_sub()

T_minus_S = tree_sub(T, S)
check_if_tree_is_structured_correctly(T_minus_S)
err_tree_sub = np.linalg.norm(flatten(T_minus_S) - (flatten(T) - flatten(S)))
print('err_tree_sub=', err_tree_sub)

# tree_mult()

T_times_S = tree_mult(T, S)
check_if_tree_is_structured_correctly(T_times_S)
err_tree_mult = np.linalg.norm(flatten(T_times_S) - (flatten(T) * flatten(S)))
print('err_tree_mult=', err_tree_mult)

# tree_scale()

c = np.random.randn()
T_times_c = tree_scale(T, c)
check_if_tree_is_structured_correctly(T_times_c)
err_tree_scale = np.linalg.norm(flatten(T_times_c) - (c * flatten(T)))
print('err_tree_scale=', err_tree_scale)

# tree_power()

p = 3
T_to_the_p = tree_power(T, p)
check_if_tree_is_structured_correctly(T_to_the_p)
err_tree_power = np.linalg.norm(flatten(T_to_the_p) - flatten(T)**p)
print('err_tree_power=', err_tree_power)

# tree_abs()

abs_T = tree_abs(T)
check_if_tree_is_structured_correctly(abs_T)
err_tree_abs = np.linalg.norm(flatten(abs_T) - np.abs(flatten(T)))
print('err_tree_abs=', err_tree_abs)

# tree_elementwise_inverse()

iT_true = make_tree(1.0/t1, 1.0/t2, 1.0/t3, 1.0/t4, 1.0/t5)
iT = tree_elementwise_inverse(T)
err_tree_elementwise_inverse = np.linalg.norm(flatten(iT_true) - flatten(iT))
print('err_tree_elementwise_inverse=', err_tree_elementwise_inverse)

# tree_sum()

sum_T = tree_sum(T)
err_tree_sum = np.abs(sum_T - np.sum(flatten(T)))
print('err_tree_sum=', err_tree_sum)

# tree_dot()

T_dot_S = tree_dot(T, S)
err_tree_dot = np.abs(T_dot_S - np.dot(flatten(T), flatten(S)))
print('err_tree_dot=', err_tree_dot)

# tree_normsquared()

normsq_T = tree_normsquared(T)
err_tree_normsquared = np.abs(normsq_T - np.linalg.norm(flatten(T))**2)
print('err_tree_normsquared=', err_tree_normsquared)

# tree_norm()

norm_T = tree_norm(T)
err_tree_norm = np.abs(norm_T - np.linalg.norm(flatten(T)))
print('err_tree_norm=', err_tree_norm)

# tree_all()

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

tree_all_test1_passed = (tree_all(T_true) == True)
tree_all_test2_passed = (tree_all(T_mixed) == False)
tree_all_test3_passed = (tree_all(T_false) == False)
print('tree_all_test1_passed=', tree_all_test1_passed)
print('tree_all_test2_passed=', tree_all_test2_passed)
print('tree_all_test3_passed=', tree_all_test3_passed)

# tree_any()

tree_any_test1_passed = (tree_all(T_true) == True)
tree_any_test2_passed = (tree_any(T_mixed) == True)
tree_any_test3_passed = (tree_any(T_false) == False)
print('tree_any_test1_passed=', tree_any_test1_passed)
print('tree_any_test2_passed=', tree_any_test2_passed)
print('tree_any_test3_passed=', tree_any_test3_passed)

# tree_eq()

T_equals_S = tree_eq(T, S)
check_if_tree_is_structured_correctly(T_equals_S)
tree_eq_test1_passed = np.all(flatten(T_equals_S) == (flatten(T) == flatten(S)))
print('tree_eq_test1_passed=', tree_eq_test1_passed)

T_equals_T = tree_eq(T, T)
check_if_tree_is_structured_correctly(T_equals_T)
tree_eq_test2_passed = np.all(flatten(T_equals_T) == (flatten(T) == flatten(T)))
print('tree_eq_test2_passed=', tree_eq_test2_passed)

t2b = t2.copy()
t2b[0,1] += 0.5
t4b = t4 - 1.8

Tb = make_tree(t1, t2b, t3, t4b, t5)
check_if_tree_is_structured_correctly(Tb)

T_equals_Tb = tree_eq(T, Tb)
check_if_tree_is_structured_correctly(T_equals_Tb)
tree_eq_test3_passed = np.all(flatten(T_equals_Tb) == (flatten(T) == flatten(Tb)))
print('tree_eq_test3_passed=', tree_eq_test3_passed)

# tree_lt()

T_lt_S = tree_lt(T, S)
check_if_tree_is_structured_correctly(T_lt_S)
tree_lt_test1_passed = np.all(flatten(T_lt_S) == (flatten(T) < flatten(S)))
print('tree_lt_test1_passed=', tree_lt_test1_passed)

T_lt_T = tree_lt(T, T)
check_if_tree_is_structured_correctly(T_lt_T)
tree_lt_test2_passed = np.all(flatten(T_lt_T) == (flatten(T) < flatten(T)))
print('tree_lt_test2_passed=', tree_lt_test2_passed)

T_lt_Tb = tree_lt(T, Tb)
check_if_tree_is_structured_correctly(T_lt_Tb)
tree_lt_test3_passed = np.all(flatten(T_lt_Tb) == (flatten(T) < flatten(Tb)))
print('tree_lt_test3_passed=', tree_lt_test3_passed)

# tree_le()

T_le_S = tree_le(T, S)
check_if_tree_is_structured_correctly(T_le_S)
tree_le_test1_passed = np.all(flatten(T_le_S) == (flatten(T) <= flatten(S)))
print('tree_le_test1_passed=', tree_le_test1_passed)

T_le_T = tree_le(T, T)
check_if_tree_is_structured_correctly(T_le_T)
tree_le_test2_passed = np.all(flatten(T_le_T) == (flatten(T) <= flatten(T)))
print('tree_le_test2_passed=', tree_le_test2_passed)

T_le_Tb = tree_le(T, Tb)
check_if_tree_is_structured_correctly(T_le_Tb)
tree_le_test3_passed = np.all(flatten(T_le_Tb) == (flatten(T) <= flatten(Tb)))
print('tree_le_test3_passed=', tree_le_test3_passed)

# tree_gt()

T_gt_S = tree_gt(T, S)
check_if_tree_is_structured_correctly(T_gt_S)
tree_gt_test1_passed = np.all(flatten(T_gt_S) == (flatten(T) > flatten(S)))
print('tree_gt_test1_passed=', tree_gt_test1_passed)

T_gt_T = tree_gt(T, T)
check_if_tree_is_structured_correctly(T_gt_T)
tree_gt_test2_passed = np.all(flatten(T_gt_T) == (flatten(T) > flatten(T)))
print('tree_gt_test2_passed=', tree_gt_test2_passed)

T_gt_Tb = tree_gt(T, Tb)
check_if_tree_is_structured_correctly(T_gt_Tb)
tree_gt_test3_passed = np.all(flatten(T_gt_Tb) == (flatten(T) > flatten(Tb)))
print('tree_gt_test3_passed=', tree_gt_test3_passed)

# tree_ge()

T_ge_S = tree_ge(T, S)
check_if_tree_is_structured_correctly(T_ge_S)
tree_ge_test1_passed = np.all(flatten(T_ge_S) == (flatten(T) >= flatten(S)))
print('tree_ge_test1_passed=', tree_ge_test1_passed)

T_ge_T = tree_ge(T, T)
check_if_tree_is_structured_correctly(T_ge_T)
tree_ge_test2_passed = np.all(flatten(T_ge_T) == (flatten(T) >= flatten(T)))
print('tree_ge_test2_passed=', tree_ge_test2_passed)

T_ge_Tb = tree_ge(T, Tb)
check_if_tree_is_structured_correctly(T_ge_Tb)
tree_ge_test3_passed = np.all(flatten(T_ge_Tb) == (flatten(T) >= flatten(Tb)))
print('tree_ge_test3_passed=', tree_ge_test3_passed)

# tree_eq_scalar()

c = t2[1,1]
T_eq_c = tree_eq_scalar(T,c)
check_if_tree_is_structured_correctly(T_eq_c)
tree_eq_scalar_test1_passed = np.all(flatten(T_eq_c) == (flatten(T) == c))
print('tree_eq_scalar_test1_passed=', tree_eq_scalar_test1_passed)

c = t1
T_eq_c = tree_eq_scalar(T,c)
check_if_tree_is_structured_correctly(T_eq_c)
tree_eq_scalar_test2_passed = np.all(flatten(T_eq_c) == (flatten(T) == c))
print('tree_eq_scalar_test2_passed=', tree_eq_scalar_test2_passed)

# tree_lt_scalar()

c = t2[1,1]
T_lt_c = tree_lt_scalar(T,c)
check_if_tree_is_structured_correctly(T_lt_c)
tree_lt_scalar_test1_passed = np.all(flatten(T_lt_c) == (flatten(T) < c))
print('tree_lt_scalar_test1_passed=', tree_lt_scalar_test1_passed)

c = t1
T_lt_c = tree_lt_scalar(T,c)
check_if_tree_is_structured_correctly(T_lt_c)
tree_lt_scalar_test2_passed = np.all(flatten(T_lt_c) == (flatten(T) < c))
print('tree_lt_scalar_test2_passed=', tree_lt_scalar_test2_passed)

# tree_le_scalar()

c = t2[1,1]
T_le_c = tree_le_scalar(T,c)
check_if_tree_is_structured_correctly(T_le_c)
tree_le_scalar_test1_passed = np.all(flatten(T_le_c) == (flatten(T) <= c))
print('tree_le_scalar_test1_passed=', tree_le_scalar_test1_passed)

c = t1
T_le_c = tree_le_scalar(T,c)
check_if_tree_is_structured_correctly(T_le_c)
tree_le_scalar_test2_passed = np.all(flatten(T_le_c) == (flatten(T) <= c))
print('tree_le_scalar_test2_passed=', tree_le_scalar_test2_passed)

# tree_gt_scalar()

c = t2[1,1]
T_gt_c = tree_gt_scalar(T,c)
check_if_tree_is_structured_correctly(T_gt_c)
tree_gt_scalar_test1_passed = np.all(flatten(T_gt_c) == (flatten(T) > c))
print('tree_gt_scalar_test1_passed=', tree_gt_scalar_test1_passed)

c = t1
T_gt_c = tree_gt_scalar(T,c)
check_if_tree_is_structured_correctly(T_gt_c)
tree_gt_scalar_test2_passed = np.all(flatten(T_gt_c) == (flatten(T) > c))
print('tree_gt_scalar_test2_passed=', tree_gt_scalar_test2_passed)

# tree_ge_scalar()

c = t2[1,1]
T_ge_c = tree_ge_scalar(T,c)
check_if_tree_is_structured_correctly(T_ge_c)
tree_ge_scalar_test1_passed = np.all(flatten(T_ge_c) == (flatten(T) >= c))
print('tree_ge_scalar_test1_passed=', tree_ge_scalar_test1_passed)

c = t1
T_ge_c = tree_ge_scalar(T,c)
check_if_tree_is_structured_correctly(T_ge_c)
tree_ge_scalar_test2_passed = np.all(flatten(T_ge_c) == (flatten(T) >= c))
print('tree_ge_scalar_test2_passed=', tree_ge_scalar_test2_passed)
