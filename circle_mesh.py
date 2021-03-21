import mshr
import dolfin as dl


def circle_mesh(center, radius, mesh_h):
    # https://github.com/NickAlger/helper_functions/blob/master/circle_mesh.ipynb
    outer_circle = mshr.Circle(dl.Point(center[0], center[1]), radius)
    mesh = mshr.generate_mesh(outer_circle, 1./mesh_h)
    return mesh