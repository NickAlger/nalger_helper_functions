import fenics
import numpy as np
from scipy.interpolate import interpn

run_test = False

class FenicsFunctionFastGridEvaluator:
    def __init__(me, fenics_function_f, oversampling_parameter=2):
        f = fenics_function_f
        V = f.function_space()
        d = V.mesh().geometric_dimension()
        mesh = V.mesh()
        hmin = mesh.hmin()
        X = V.tabulate_dof_coordinates()

        f.set_allow_extrapolation(True)

        min_point = np.min(X, axis=0)
        max_point = np.max(X, axis=0)

        output_shape = f(np.zeros(d)).shape

        nn = (oversampling_parameter * (max_point - min_point) / hmin).astype(int)
        me.xx = tuple([np.linspace(min_point[i], max_point[i], nn[i]) for i in range(d)])
        XX = np.meshgrid(*me.xx, indexing='ij')
        pp = np.array([X.reshape(-1) for X in XX]).T

        N = np.prod(nn)
        ff = []
        for k in range(N):
            ff.append(f(pp[k, :]))
        me.F = np.array(ff).reshape(tuple(nn) + output_shape)

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

    t = time()
    eval_f = FenicsFunctionFastGridEvaluator(f)
    dt_construct_grid_evaluator = time() - t
    print('dt_construct_grid_evaluator=', dt_construct_grid_evaluator)

    n_test = int(1e4)
    qq = np.random.rand(n_test, d)
    qq = qq[np.linalg.norm(qq - 0.5, axis=1) < 0.5]
    n_test = qq.shape[0]
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


    #### Test vector function space ####

    V_vec = fenics.VectorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree())
    f = fenics.Function(V_vec)

    f0 = fenics.interpolate(fenics.Expression('sin(x[0]) + cos(x[1])', element=V.ufl_element()), V)
    f1 = fenics.interpolate(fenics.Expression('cos(x[0]) + sin(x[1])', element=V.ufl_element()), V)
    fenics.assign(f.sub(0), f0)
    fenics.assign(f.sub(1), f1)
    f.set_allow_extrapolation(True)

    t = time()
    eval_f = FenicsFunctionFastGridEvaluator(f)
    dt_construct_grid_evaluator = time() - t
    print('dt_construct_grid_evaluator=', dt_construct_grid_evaluator)

    n_test = int(1e4)
    qq = np.random.rand(n_test, d)
    qq = qq[np.linalg.norm(qq - 0.5, axis=1) < 0.5]
    n_test = qq.shape[0]
    t = time()
    zz = eval_f(qq)
    dt_interpn_vec = time() - t
    print('n_test=', n_test, ', dt_interpn_vec=', dt_interpn_vec)

    t = time()
    zz2 = np.zeros((n_test,2))
    for k in range(n_test):
        zz2[k,:] = f(qq[k,:])
    dt_interpn_vec_loop = time() - t
    print('n_test=', n_test, ', dt_interpn_vec_loop=', dt_interpn_vec_loop)

    err_grid_interpolation_vec = np.linalg.norm(zz2 - zz)/np.linalg.norm(zz2)
    print('err_grid_interpolation_vec=', err_grid_interpolation_vec)


    #### Test Tensor Function Space ####

    V_mat = fenics.TensorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree())
    f = fenics.Function(V_mat)

    f0 = fenics.interpolate(fenics.Expression('sin(x[0]) + cos(x[1])', element=V.ufl_element()), V)
    f1 = fenics.interpolate(fenics.Expression('cos(x[0]) + sin(x[1])', element=V.ufl_element()), V)
    f2 = fenics.interpolate(fenics.Expression('pow(x[1],2) + x[0]', element=V.ufl_element()), V)
    f3 = fenics.interpolate(fenics.Expression('x[0] * x[1]', element=V.ufl_element()), V)
    fenics.assign(f.sub(0), f0)
    fenics.assign(f.sub(1), f1)
    fenics.assign(f.sub(2), f2)
    fenics.assign(f.sub(3), f3)
    f.set_allow_extrapolation(True)

    t = time()
    eval_f = FenicsFunctionFastGridEvaluator(f)
    dt_construct_grid_evaluator = time() - t
    print('dt_construct_grid_evaluator=', dt_construct_grid_evaluator)

    n_test = int(1e4)
    qq = np.random.rand(n_test, d)
    qq = qq[np.linalg.norm(qq - 0.5, axis=1) < 0.5]
    n_test = qq.shape[0]
    t = time()
    zz = eval_f(qq)
    dt_interpn_mat = time() - t
    print('n_test=', n_test, ', dt_interpn_mat=', dt_interpn_mat)

    t = time()
    zz2 = np.zeros((n_test, 4))
    for k in range(n_test):
        zz2[k, :] = f(qq[k, :])
    dt_interpn_mat_loop = time() - t
    print('n_test=', n_test, ', dt_interpn_mat_loop=', dt_interpn_mat_loop)

    err_grid_interpolation_mat = np.linalg.norm(zz2 - zz) / np.linalg.norm(zz2)
    print('err_grid_interpolation_mat=', err_grid_interpolation_mat)
