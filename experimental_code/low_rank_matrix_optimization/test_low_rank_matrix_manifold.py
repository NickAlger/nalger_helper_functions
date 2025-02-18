import numpy as np
import jax

from nalger_helper_functions.low_rank_matrix_optimization.low_rank_matrix_manifold import *
import nalger_helper_functions.tree_linalg as tla

jax.config.update("jax_enable_x64", True) # enable double precision

# Test low_rank_to_full()

N = 17
r = 5
M = 12

X = np.random.randn(N,r)
Y = np.random.randn(r,M)
base = (X, Y)

A = low_rank_to_full(base)
A_true = X @ Y
err_low_rank_to_full = np.linalg.norm(A - A_true) / np.linalg.norm(A_true)
print('err_low_rank_to_full=', err_low_rank_to_full)


# test left_orthogonalize_low_rank()

Q, Y2 = left_orthogonalize_low_rank(base)

A2 = Q @ Y2
err_mult_base = np.linalg.norm(A2 - A_true) / np.linalg.norm(A_true)
print('err_mult_base=', err_mult_base)

err_orth_left_orthogonalize_low_rank = np.linalg.norm(Q.T @ Q - np.eye(r))
print('err_orth_left_orthogonalize_low_rank=', err_orth_left_orthogonalize_low_rank)


# Test tangent_vector_to_full()

dX = np.random.randn(N,r)
dY = np.random.randn(r,M)
perturbation = (dX, dY)

s = 1e-7
v_diff = (low_rank_to_full((X + s*dX, Y + s*dY)) - low_rank_to_full((X, Y))) / s
v = tangent_vector_to_full(base, perturbation)
err_tangent_vector_to_full = np.linalg.norm(v - v_diff) / np.linalg.norm(v_diff)
print('s=', s, ', err_tangent_vector_to_full=', err_tangent_vector_to_full)


# Test add_sequences()

perturbation1 = perturbation

dX2 = np.random.randn(N,r)
dY2 = np.random.randn(r,M)
perturbation2 = (dX, dY)

perturbation12 = tla.add(perturbation1, perturbation2)

v1 = tangent_vector_to_full(base, perturbation1)
v2 = tangent_vector_to_full(base, perturbation2)
v12_true = v1 + v2

v12 = tangent_vector_to_full(base, perturbation12)
err_add_sequences = np.linalg.norm(v12 - v12_true) / np.linalg.norm(v12_true)
print('err_add_sequences=', err_add_sequences)


# Test scale_sequence()

c = np.random.randn()

scaled_perturbation = tla.scale(perturbation, c)

cv = tangent_vector_to_full(base, scaled_perturbation)
cv_true = c * tangent_vector_to_full(base, perturbation)
err_scale_sequence = np.linalg.norm(cv - cv_true) / np.linalg.norm(cv_true)
print('err_scale_sequence=', err_scale_sequence)


# Test tangent_oblique_projection()

left_orthogonal_base = left_orthogonalize_low_rank(base)
standard_perturbation = tangent_oblique_projection(left_orthogonal_base, perturbation)

v_true = tangent_vector_to_full(left_orthogonal_base, perturbation)
v = tangent_vector_to_full(left_orthogonal_base, standard_perturbation)
err_mult_tangent_oblique_projection = np.linalg.norm(v - v_true) / np.linalg.norm(v)
print('err_mult_tangent_oblique_projection=', err_mult_tangent_oblique_projection)

Q, _ = left_orthogonal_base
dX_perp, _ = standard_perturbation

err_perp_tangent_oblique_projection = np.linalg.norm(Q.T @ dX_perp)
print('err_perp_tangent_oblique_projection=', err_perp_tangent_oblique_projection)


# Test tangent_oblique_projection_transpose()

p = perturbation1
q = perturbation2
Fp = tangent_oblique_projection(left_orthogonal_base, p)
FTq = tangent_oblique_projection_transpose(left_orthogonal_base, q)

t1 = tla.dot(Fp, q)
t2 = tla.dot(p, FTq)

err_tangent_oblique_projection_transpose = np.abs(t1 - t2) / np.abs(t1 + t2)
print('err_tangent_oblique_projection_transpose=', err_tangent_oblique_projection_transpose)


# Test inner_product_of_tangent_vectors()

standard_perturbation1 = tangent_oblique_projection(left_orthogonal_base, perturbation1)
standard_perturbation2 = tangent_oblique_projection(left_orthogonal_base, perturbation2)

inner_product_helper_matrix = make_tangent_mass_matrix_helper(left_orthogonal_base)

IP = inner_product_of_tangent_vectors(standard_perturbation1, standard_perturbation2, inner_product_helper_matrix)

v1 = tangent_vector_to_full(left_orthogonal_base, standard_perturbation1)
v2 = tangent_vector_to_full(left_orthogonal_base, standard_perturbation2)
IP_true = np.sum(v1 * v2)

err_inner_product_of_tangent_vectors = np.linalg.norm(IP - IP_true) / np.linalg.norm(IP_true)
print('err_inner_product_of_tangent_vectors=', err_inner_product_of_tangent_vectors)


# Test attached_tangent_vector_as_low_rank()

v1 = low_rank_to_full(base) + tangent_vector_to_full(base, perturbation)
v2 = low_rank_to_full(attached_tangent_vector_as_low_rank(base, perturbation))

err_attached_tangent_vector_as_low_rank = np.linalg.norm(v2-v1) / np.linalg.norm(v1)
print('err_attached_tangent_vector_as_low_rank=', err_attached_tangent_vector_as_low_rank)


# Test add_low_rank_matrices()

M1 = (np.random.randn(N,3), np.random.randn(3,M))
M2 = (np.random.randn(N,2), np.random.randn(2,M))
M3 = (np.random.randn(N,5), np.random.randn(5,M))
M4 = (np.random.randn(N,1), np.random.randn(1,M))

M = low_rank_to_full(M1) + low_rank_to_full(M2) + low_rank_to_full(M3) + low_rank_to_full(M4)

MM = [M1, M2, M3, M4]
M2 = low_rank_to_full(add_low_rank_matrices(MM))

err_add_low_rank_matrices = np.linalg.norm(M2 - M) / np.linalg.norm(M)
print('err_add_low_rank_matrices=', err_add_low_rank_matrices)


# Test scale_low_rank_matrix()

c = 0.5432
cM1_full = low_rank_to_full(scale_low_rank_matrix(M1, c))
cM1_full_true = c * low_rank_to_full(M1)

err_scale_low_rank_matrix = np.linalg.norm(cM1_full - cM1_full_true) / np.linalg.norm(cM1_full_true)
print('err_scale_low_rank_matrix=', err_scale_low_rank_matrix)


# Test retract_tangent_vector()

new_rank = base[0].shape[1] + 3

retracted_vector = retract_tangent_vector(base, perturbation, new_rank)
v = low_rank_to_full(retracted_vector)

U, ss, Vt = np.linalg.svd(low_rank_to_full(base) + tangent_vector_to_full(base, perturbation))
v_true = U[:,:new_rank] @ np.diag(ss[:new_rank]) @ Vt[:new_rank,:]

err_retract_vector = np.linalg.norm(v - v_true) / np.linalg.norm(v_true)
print('err_retract_vector=', err_retract_vector)

# Test commutativity of oblique projection and application of mass matrix

M_Pi_p = apply_tangent_mass_matrix(tangent_oblique_projection(left_orthogonal_base, perturbation), inner_product_helper_matrix)
Pi_M_p = tangent_oblique_projection(left_orthogonal_base, apply_tangent_mass_matrix(perturbation, inner_product_helper_matrix))

non_commutativity_obliqueproject_vs_massmatrix = tla.norm(tla.sub(M_Pi_p, Pi_M_p)) / tla.norm(tla.add(M_Pi_p, Pi_M_p))
print('non_commutativity_obliqueproject_vs_massmatrix=', non_commutativity_obliqueproject_vs_massmatrix, ' (should be nonzero)')

# Test commutativity of orth projection and application of mass matrix

M_Pi_p = apply_tangent_mass_matrix(tangent_orthogonal_projection(left_orthogonal_base, perturbation), inner_product_helper_matrix)
Pi_M_p = tangent_orthogonal_projection(left_orthogonal_base, apply_tangent_mass_matrix(perturbation, inner_product_helper_matrix))

non_commutativity_orthproject_vs_massmatrix = tla.norm(tla.sub(M_Pi_p, Pi_M_p)) / tla.norm(tla.add(M_Pi_p, Pi_M_p))
print('non_commutativity_orthproject_vs_massmatrix=', non_commutativity_orthproject_vs_massmatrix)




