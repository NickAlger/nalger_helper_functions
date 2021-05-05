import numpy as np
import dolfin as dl


def dlfct2array(scalar_or_vector_or_tensor_function_u):
    u = scalar_or_vector_or_tensor_function_u

    val_shape = tuple(u.value_shape())
    m = int(np.prod(val_shape))

    if m == 1:
        U = u.vector()[:]
    else:
        N = int(u.function_space().dim() / m)

        U = np.zeros((m, N))
        for k in range(m):
            k_dofs = u.function_space().sub(k).dofmap().dofs()
            U[k, :] = u.vector()[k_dofs]

        U = U.T.reshape((N,) + val_shape)
        if m == 1:
            U = U.reshape(-1)

    return U


def array2dlfct(array_U, scalar_vector_or_tensor_output_function_space_VV):
    U = array_U
    VV = scalar_vector_or_tensor_output_function_space_VV

    val_shape = U.shape[1:]
    m = int(np.prod(val_shape))

    if m == 1:
        u = dl.Function(VV)
        u.vector()[:] = array_U.reshape(-1)
    else:
        U = U.reshape((-1, m))
        u = dl.Function(VV)
        for k in range(m):
            k_dofs = VV.sub(k).dofmap().dofs()
            u.vector()[k_dofs] = U[:, k].reshape(-1).copy()

    return u