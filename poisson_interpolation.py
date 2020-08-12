import numpy as np
import fenics
import scipy.linalg as sla
from poisson_point_source_solver import NeumannPoissonSolver

run_test = False

class PoissonSquaredInterpolation:
    def __init__(me, function_space_V, point_inds_ii=None):
        me.V = function_space_V
        me.N = V.dim()

        u_trial = fenics.TrialFunction(V)
        v_test = fenics.TestFunction(V)

        mass_form = u_trial * v_test * fenics.dx
        me.M = fenics.assemble(mass_form)

        me.constant_one_function = fenics.interpolate(fenics.Constant(1.0), me.V)

        me.NPPSS = NeumannPoissonSolver(me.V)
        me.solve_neumann_poisson = me.NPPSS.solve
        me.solve_neumann_point_source = me.NPPSS.solve_point_source

        me.ii = list() # point inds
        me.impulse_responses = list()
        me.solve_S = lambda x: np.nan
        me.eta = np.zeros(me.num_pts)
        me.mu = np.nan
        me.smooth_basis = list()
        me.weighting_functions = list()

        if point_inds_ii is not None:
            me.add_points(point_inds_ii)

    def add_points(me, new_point_inds_ii):
        num_old_pts = me.num_pts
        num_new_pts = len(new_point_inds_ii)

        me.ii = me.ii + list(new_point_inds_ii)

        new_impulse_responses = list()
        for k in range(num_new_pts):
            ind = me.ii[num_old_pts + k]
            new_impulse_responses.append(me.solve_neumann_point_source(ind).vector())
        me.impulse_responses = me.impulse_responses + new_impulse_responses

        S = np.zeros((me.num_pts, me.num_pts))
        for i in range(me.num_pts):
            for j in range(me.num_pts):
                S[i, j] = me.impulse_responses[i].inner(me.M * me.impulse_responses[j])
        me.solve_S = make_dense_lu_solver(S)
        me.eta = me.solve_S(np.ones(me.num_pts))
        me.mu = np.dot(np.ones(me.num_pts), me.eta)

        new_smooth_basis_vectors = list()
        for k in range(num_new_pts):
            qk = me.M * new_impulse_responses[k]
            new_smooth_basis_vectors.append(-me.solve_neumann_poisson(qk).vector())
        me.smooth_basis = me.smooth_basis + new_smooth_basis_vectors

        me.compute_weighting_functions()

    def compute_weighting_functions(me):
        me.weighting_functions = list()
        I = np.eye(me.num_pts)
        for k in range(me.num_pts):
            ek = I[:,k]
            me.weighting_functions.append(me.interpolate_values(ek))

    @property
    def num_pts(me):
        return len(me.ii)

    def interpolate_values(me, values_at_points_y):
        y = values_at_points_y
        alpha = (1. / me.mu) * np.dot(me.eta, y)
        p = -me.solve_S(y - alpha * np.ones(me.num_pts))
        u = alpha * me.constant_one_function.vector()
        for k in range(len(me.smooth_basis)):
            u = u + me.smooth_basis[k] * p[k]
        return u


def make_dense_lu_solver(M):
    M_lu, M_pivot = sla.lu_factor(M)
    solve_M = lambda b: sla.lu_solve((M_lu, M_pivot), b)
    return solve_M


if run_test:
    import matplotlib.pyplot as plt
    from matplotlib_mouse_click import Click

    n = 75
    num_pts0 = 51

    mesh = fenics.UnitSquareMesh(n, n)
    V = fenics.FunctionSpace(mesh, 'CG', 2)

    def nearest_ind(q, xx):
        return np.argmin(np.linalg.norm(q - xx, axis=1))

    def choose_random_mesh_nodes(V, num_pts0):
        np.random.seed(0)
        xx0 = np.random.rand(num_pts0, mesh.geometric_dimension()) * 0.7 + 0.15
        coords = V.tabulate_dof_coordinates()
        closest_inds = []
        for k in range(xx0.shape[0]):
            closest_inds.append(nearest_ind(xx0[k, :], coords))

        point_inds_ii = np.unique(closest_inds)
        xx = coords[point_inds_ii, :]
        num_pts = xx.shape[0]
        return point_inds_ii, xx, num_pts

    point_inds_ii, xx, num_pts = choose_random_mesh_nodes(V, num_pts0)
    y = np.random.randn(num_pts)

    PSI = PoissonSquaredInterpolation(V, point_inds_ii=point_inds_ii)

    y = np.random.randn(PSI.num_pts)
    u = PSI.interpolate_values(y)

    u_fct = fenics.Function(V)
    u_fct.vector().set_local(u)

    plt.figure()
    c = fenics.plot(u_fct)
    for k in range(xx.shape[0]):
        plt.plot(xx[k, 0], xx[k, 1], '.r')
    plt.colorbar(c)
    plt.title('interpolation of random data')

    #

    def plot_kth_weighting_function(k):
        wk = fenics.Function(V)
        wk.vector().set_local(PSI.weighting_functions[k])
        fenics.plot(wk)
        for k in range(xx.shape[0]):
            plt.plot(xx[k, 0], xx[k, 1], '.r')

    fig = plt.figure()
    plot_kth_weighting_function(0)
    plt.title('weighting function (click on dot)')

    def onclick(event, ax):
        q = np.array([event.xdata, event.ydata])
        k = nearest_ind(q, xx)
        plot_kth_weighting_function(k)

    click = Click(plt.gca(), onclick, button=1)
