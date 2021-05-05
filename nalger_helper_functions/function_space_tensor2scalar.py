import dolfin as dl


def function_space_tensor2scalar(VV):
    return dl.FunctionSpace(VV.mesh(), VV.ufl_element().family(), VV.ufl_element().degree())


def function_space_scalar2tensor(V, val_shape):
    if len(val_shape) == 0:
        VV = V
    elif len(val_shape) == 1:
        VV = dl.VectorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree(), dim=val_shape[0])
    else:
        VV = dl.TensorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree(), shape=val_shape)
    return VV