import numpy as np


def box_mesh_lexsort(function_space_V):
    # https://github.com/NickAlger/helper_functions/blob/master/box_mesh_lexsort.ipynb
    coords = function_space_V.tabulate_dof_coordinates()
    lexsort_inds = np.lexsort(coords[:,::-1].T)
    return lexsort_inds