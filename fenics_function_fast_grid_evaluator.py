import fenics
import numpy as np
from scipy.interpolate import interpn

run_test = False

class FenicsFunctionFastGridEvaluator:
    def __init__(me, f_vec, V, oversampling_parameter=2):
        d = V.mesh().geometric_dimension()
        mesh = V.mesh()
        hmin = mesh.hmin()
        X = V.tabulate_dof_coordinates()

        f = fenics.Function(V)
        f.vector()[:] = f_vec.copy()
        f.set_allow_extrapolation(True)

        min_point = np.min(X, axis=0)
        max_point = np.max(X, axis=0)
        nn = (oversampling_parameter * (max_point - min_point) / hmin).astype(int)
        me.xx = tuple([np.linspace(min_point[i], max_point[i], nn[i]) for i in range(d)])
        XX = np.meshgrid(*me.xx, indexing='ij')
        pp = np.array([X.reshape(-1) for X in XX]).T

        ff = np.zeros(nn).reshape(-1)
        for k in range(len(ff)):
            ff[k] = f(pp[k, :])
        me.F = ff.reshape(nn)


    def __call__(me, points_pp):
        return interpn(me.xx, me.F, points_pp, bounds_error=False, fill_value=0.0)


if run_test:
    import mshr
    from time import time
    mesh_h = 1e-2

    n = 1. / mesh_h

    outer_circle = mshr.Circle(fenics.Point(0.5, 0.5), 0.5)
    mesh = mshr.generate_mesh(outer_circle, n)

    V = fenics.FunctionSpace(mesh, 'CG', 1)
    d = mesh.geometric_dimension()

    f = fenics.interpolate(fenics.Expression('sin(x[0]) + cos(x[1])', element=V.ufl_element()), V)
    f.set_allow_extrapolation(True)
    f_vec = f.vector()[:]

    t = time()
    eval_f = FenicsFunctionFastGridEvaluator(f_vec, V)
    dt_construct_grid_evaluator = time() - t
    print('dt_construct_grid_evaluator=', dt_construct_grid_evaluator)

    n_test = int(1e4)
    qq = np.random.rand(n_test, d)
    t = time()
    zz = eval_f(qq)
    dt_interpn = time() - t
    print('n_test=', n_test, ', dt_interpn=', dt_interpn)

    t = time()
    zz2 = np.zeros(n_test)
    for k in range(n_test):
        zz2[k] = f(qq[k,:])
    dt_interpn_loop = time() - t
    print('n_test=', n_test, ', dt_interpn_loop=', dt_interpn_loop)

    err_grid_interpolation = np.linalg.norm(zz2 - zz)/np.linalg.norm(zz2)
    print('err_grid_interpolation=', err_grid_interpolation)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(qq[:,0], qq[:,1], c=zz2)
    plt.title('zz2')

    plt.figure()
    plt.scatter(qq[:, 0], qq[:, 1], c=zz)
    plt.title('zz')
