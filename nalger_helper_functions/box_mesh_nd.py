import numpy as np
import dolfin as dl


def box_mesh_nd(box_min, box_max, grid_shape):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/blob/master/jupyter_notebooks/box_mesh.ipynb
    d = len(box_min)
    if d == 1:
        mesh = dl.IntervalMesh(grid_shape[0] - 1, box_min[0], box_max[0])
    elif d == 2:
        mesh = dl.RectangleMesh(dl.Point(box_min), dl.Point(box_max), grid_shape[0] - 1, grid_shape[1] - 1)
    elif d == 3:
        mesh = dl.BoxMesh(dl.Point(box_min), dl.Point(box_max), grid_shape[0] - 1, grid_shape[1] - 1, grid_shape[2] - 1)
    else:
        raise RuntimeError('grid_mesh only supports d=1,2,3')
    return mesh