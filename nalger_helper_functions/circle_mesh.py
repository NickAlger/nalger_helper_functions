import mshr
import dolfin as dl


def circle_mesh(center, radius, mesh_h):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/blob/master/jupyter_notebooks/circle_mesh.ipynb
    outer_circle = mshr.Circle(dl.Point(center[0], center[1]), radius)
    mesh = mshr.generate_mesh(outer_circle, 1./mesh_h)
    return mesh