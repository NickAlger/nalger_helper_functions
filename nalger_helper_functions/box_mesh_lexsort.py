import numpy as np


def box_mesh_lexsort(function_space_V):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/blob/master/jupyter_notebooks/box_mesh_lexsort.ipynb
    coords = function_space_V.tabulate_dof_coordinates()
    lexsort_inds = np.lexsort(coords[:,::-1].T)
    return lexsort_inds