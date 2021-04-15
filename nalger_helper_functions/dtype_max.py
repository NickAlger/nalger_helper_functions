import numpy as np


def dtype_max(list_of_dtypes):
    return np.sum([np.array([1], dtype=dtype) for dtype in list_of_dtypes]).dtype