import numpy as np

# Old partitioning and ordering

p1 = [1,2,3,4,5]
p2 = [6,7,8,9,10]
p3 = [11,12,13,14,15]

# New partitioning and ordering

o1 = [1,2,3,5,6,7]
o2 = [4,8,12]
o3 = [9,10,11,13,14,15]

# Expanded new partitions

o1e = [1,2,3,4,5,6,7,8,10,11]
o2e = [3,4,7,8,12]
o3e = [5,6,9,10,11,12,13,14,15]

# Convert from 1-based indexing to 0=based indexing

def one_index_to_zero_index(x):
    return list(np.array(x) - 1)

p1 = one_index_to_zero_index(p1)
p2 = one_index_to_zero_index(p2)
p3 = one_index_to_zero_index(p3)

o1 = one_index_to_zero_index(o1)
o2 = one_index_to_zero_index(o2)
o3 = one_index_to_zero_index(o3)

o1e = one_index_to_zero_index(o1e)
o2e = one_index_to_zero_index(o2e)
o3e = one_index_to_zero_index(o3e)

# Blocks in new partitioning

B1 = np.arange(6*10).reshape(10,6).T # column-first
B2 = np.arange(3*5).reshape(5,3).T
B3 = np.arange(6*9).reshape(9,6).T

print('B1:')
print(B1)

print('B2:')
print(B2)

print('B3:')
print(B3)

# Transfer vector in old ordering to new ordering. Also, do padded version.

x_old_parts = [np.arange(0,5)+1, np.arange(5,10)+1, np.arange(10,15)+1]
x_old = np.concatenate(x_old_parts)

x_new_parts = [x_old[o1], x_old[o2], x_old[o3]]
x_new_padded_parts = [x_old[o1e], x_old[o2e], x_old[o3e]]

print('x_old_parts:')
print(x_old_parts)

print('x_new_parts:')
print(x_new_parts)

print('x_new_padded_parts:')
print(x_new_padded_parts)

# Perform block row dense matvec in new ordering

y_new_parts = [Bk @ xk for Bk, xk in zip([B1, B2, B3], x_new_padded_parts)]

print('y_new_parts:')
print(y_new_parts)

# Convert result to old ordering

y_old = np.zeros(15)
y_old[o1] = y_new_parts[0]
y_old[o2] = y_new_parts[1]
y_old[o3] = y_new_parts[2]

print('y_old:')
print(y_old)

y_old_parts = [y_old[p1], y_old[p2], y_old[p3]]

print('y_old_parts:')
print(y_old_parts)

# Compare to direct dense matvec in old ordering

Bold = np.zeros((15,15))
Bold[np.ix_(o1,o1e)] = B1
Bold[np.ix_(o2,o2e)] = B2
Bold[np.ix_(o3,o3e)] = B3

y_old_true = Bold @ x_old

print('y_old_true:')
print(y_old_true)

err = np.linalg.norm(y_old_true - y_old)
print('err=', err)
