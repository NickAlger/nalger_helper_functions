import fenics
import numpy as np


class NeumannPoissonSolver:
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/neumann_poisson_solver.ipynb
    def __init__(me, function_space_V):
        me.V = function_space_V
        me.dof_coords = me.V.tabulate_dof_coordinates()

        f = fenics.Function(me.V)

        u = fenics.TrialFunction(me.V)
        v = fenics.TestFunction(me.V)
        a = fenics.inner(fenics.grad(u), fenics.grad(v)) * fenics.dx
        rhs = f * v * fenics.dx

        A = fenics.assemble(a)
        me.b = fenics.assemble(rhs)

        const_fct = fenics.Function(me.V)
        const_fct.interpolate(fenics.Constant(1.0))
        const_vec = const_fct.vector()
        me.const_vec = const_vec / fenics.norm(const_vec)

        prec = fenics.PETScPreconditioner('hypre_amg')
        fenics.PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
        me._solver = fenics.PETScKrylovSolver('cg', prec)
        me._solver.set_operator(A)

    def solve(me, rhs_fenics_vector, atol=0.0, rtol=1e-7, maxiter=100, verbose=False):
        me._solver.parameters['absolute_tolerance'] = atol
        me._solver.parameters['relative_tolerance'] = rtol
        me._solver.parameters['maximum_iterations'] = maxiter
        me._solver.parameters['monitor_convergence'] = verbose

        b = rhs_fenics_vector.copy()
        b = b - b.inner(me.const_vec) * me.const_vec

        x = fenics.Function(me.V)
        me._solver.solve(x.vector(), b)
        return x

    def solve_point_source(me, point, point_type='ind', atol=0.0, rtol=1e-10, maxiter=100, verbose=False):
        me.b.zero()
        if point_type == 'coords':
            point_fenics = fenics.Point(point)
        elif point_type == 'ind':
            point_fenics = fenics.Point(me.dof_coords[point, :])
        elif point_type == 'fenics':
            point_fenics = point
        else:
            raise RuntimeError('invalid point_type')
        ps = fenics.PointSource(me.V, point_fenics, 1.0)
        ps.apply(me.b)

        return me.solve(me.b, atol=atol, rtol=rtol, maxiter=maxiter, verbose=verbose)
