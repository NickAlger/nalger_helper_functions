import fenics

class FenicsFunctionSmoother:
    def __init__(me, function_space_V, smoothing_time=1e-2, num_timesteps=10):
        me.num_timesteps = num_timesteps
        me.dt = smoothing_time / num_timesteps

        u = fenics.TrialFunction(function_space_V)
        v = fenics.TestFunction(function_space_V)

        mass_form = u * v * fenics.dx
        stiffness_form = fenics.inner(fenics.grad(u), fenics.grad(v)) * fenics.dx

        me.M = fenics.assemble(mass_form)
        Z = fenics.assemble(mass_form + me.dt * stiffness_form)

        me.Z_solver = fenics.LUSolver(Z)

    def smooth(me, function_f):
        for k in range(me.num_timesteps):
            me.Z_solver.solve(function_f.vector(), me.M * function_f.vector())
