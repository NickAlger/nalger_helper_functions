import fenics
import matplotlib.pyplot as plt


def fenics_interactive_impulse_response_plot(apply_A, function_space_V):
    fig = plt.figure()
    c = fenics.plot(fenics.Function(function_space_V))
    plt.colorbar(c)
    plt.title('left click adds points, right click resets')
    point_source_dual_vector = fenics.assemble(fenics.Constant(0.0) * fenics.TestFunction(function_space_V) * fenics.dx)

    Adelta = fenics.Function(function_space_V)
    def onclick(event):
        delta_p = fenics.PointSource(function_space_V, fenics.Point(event.xdata, event.ydata), 1.0)
        if event.button == 3:
            point_source_dual_vector[:] = 0.
        delta_p.apply(point_source_dual_vector)
        Adelta.vector()[:] = apply_A(point_source_dual_vector)
        # Adelta = vec2fct(apply_A(point_source_dual_vector), function_space_V)
        plt.clf()
        c = fenics.plot(Adelta)
        plt.colorbar(c)
        plt.title('left click adds points, right click resets')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    return fig, cid


def vec2fct(u_vec, Vh):
    u_fct = fenics.Function(Vh)
    u_fct.vector()[:] = u_vec.copy()
    return u_fct
