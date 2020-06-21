import fenics
import matplotlib.pyplot as plt

run_test = False

def fenics_interactive_impulse_response_plot(apply_A, V):
    fig = plt.figure()
    fenics.plot(fenics.Function(V))
    plt.title('left click adds points, right click resets')
    point_source_dual_vector = fenics.assemble(fenics.Constant(0.0) * fenics.TestFunction(V) * fenics.dx)

    def onclick(event):
        delta_p = fenics.PointSource(V, fenics.Point(event.xdata, event.ydata), 1.0)
        if event.button == 3:
            point_source_dual_vector[:] = 0.
        delta_p.apply(point_source_dual_vector)
        Adelta = vec2fct(apply_A(point_source_dual_vector[:]), V)
        fenics.plot(Adelta)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    return fig, cid

def vec2fct(u_vec, Vh):
    u_fct = fenics.Function(Vh)
    u_fct.vector()[:] = u_vec.copy()
    return u_fct

if run_test:
    n = 40
    mesh = fenics.UnitSquareMesh(n,n)
    V = fenics.FunctionSpace(mesh, 'CG', 1)
    u_trial = fenics.TrialFunction(V)
    v_test = fenics.TestFunction(V)
    a = fenics.inner(fenics.grad(u_trial), fenics.grad(v_test)) * fenics.dx + u_trial * v_test * fenics.dx
    b = fenics.Constant(0.0) * v_test * fenics.dx
    A_fenics = fenics.assemble(a)
    b_fenics = fenics.assemble(b)
    x_fenics = fenics.Function(V)

    solve_A_fenics = fenics.LUSolver(A_fenics)
    def apply_invA(b_vec):
        b_fenics[:] = b_vec.copy()
        solve_A_fenics.solve(x_fenics.vector(), b_fenics)
        return x_fenics.vector()[:].copy()

    fenics_interactive_impulse_response_plot(apply_invA, V)

