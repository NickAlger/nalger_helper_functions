import fenics
import numpy as np

run_test = False

class FenicsFunctionExtendByZeroEvaluator:
    def __init__(me, V):
        me.f = fenics.Function(V)
        me.bbt = fenics.BoundingBoxTree()
        me.bbt.build(V.mesh())
        me.d = V.mesh().geometric_dimension()

    def __call__(me, f_vec, points_pp):
        me.f.vector()[:] = f_vec.copy()
        if len(points_pp.shape) == 1:
            N = 1
        else: # len(points_pp.shape) == 2
            N, _ = points_pp.shape
        points_pp = points_pp.reshape((N, me.d))

        inds_of_points_in_mesh = []
        for k in range(N):
            pk = fenics.Point(points_pp[k,:])
            if me.bbt.compute_collisions(pk):
                inds_of_points_in_mesh.append(k)

        ff = np.zeros(N)
        for k in inds_of_points_in_mesh:
            pk = fenics.Point(points_pp[k, :])
            ff[k] = me.f(pk)
        return ff


if run_test:
    mesh = fenics.UnitSquareMesh(10,10)
    V = fenics.FunctionSpace(mesh, 'CG', 2)

    V_evaluator = FenicsFunctionExtendByZeroEvaluator(V)

    u = fenics.Function(V)
    u.vector()[:] = np.random.randn(V.dim())

    p = np.array([0.5,0.5])
    up_true = u(fenics.Point(p))
    up = V_evaluator(u.vector()[:], p)
    err_V_evaluator_single = np.abs(up - up_true) / np.abs(up_true)
    print('err_V_evaluator_single=', err_V_evaluator_single)

    N = 1000
    pp = 3 * np.random.rand(N,2) - 1.0 # random points in [-1, 2]^2

    upp_true = np.zeros(N)
    for k in range(N):
        pk = pp[k,:]
        if np.all(pk >= 0) and np.all(pk <= 1):
            upp_true[k] = u(fenics.Point(pk))

    upp = V_evaluator(u.vector()[:], pp)

    err_V_evaluator_many = np.linalg.norm(upp - upp_true)/np.linalg.norm(upp_true)
    print('err_V_evaluator=', err_V_evaluator_many)
