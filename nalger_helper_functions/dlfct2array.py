import numpy as np
import dolfin as dl


def dlfct2array(scalar_or_vector_or_tensor_function_u, scalar_function_space_V=None):
    u = scalar_or_vector_or_tensor_function_u
    V = scalar_function_space_V

    u.set_allow_extrapolation(True)

    val_shape = tuple(u.value_shape())
    m = int(np.prod(val_shape))

    if m == 1:
        if V is None:
            return u.vector()[:]
        else:
            return dl.interpolate(u, V).vector()[:]
    else:
        if V is None:
            N = int(u.function_space().dim() / m)
        else:
            N = V.dim()

        U = np.zeros((m, N))
        for k in range(m):
            if V is None:
                k_dofs = u.function_space().sub(k).dofmap().dofs()
                U[k, :] = u.vector()[k_dofs]
            else:
                U[k, :] = dl.interpolate(u.sub(k), V).vector()[:]

        U = U.T.reshape((N,) + val_shape)
        if m == 1:
            U = U.reshape(-1)
        return U
