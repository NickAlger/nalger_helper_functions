import dolfin as dl
import matplotlib.pyplot as plt


def interactive_impulse_response_plot(apply_A, function_space_V, print_point=True):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/nalger_helper_functions/interactive_impulse_response_plot.py
    fig = plt.figure()
    c = dl.plot(dl.Function(function_space_V))
    plt.colorbar(c)
    plt.title('left click adds points, right click resets')
    point_source_dual_vector = dl.assemble(dl.Constant(0.0) * dl.TestFunction(function_space_V) * dl.dx)

    Adelta = dl.Function(function_space_V)
    def onclick(event):
        if print_point:
            print('(', event.xdata, ',', event.ydata, ')')
        delta_p = dl.PointSource(function_space_V, dl.Point(event.xdata, event.ydata), 1.0)
        if event.button == 3:
            point_source_dual_vector[:] = 0.
        delta_p.apply(point_source_dual_vector)
        Adelta.vector()[:] = apply_A(point_source_dual_vector)
        plt.clf()
        c = dl.plot(Adelta)
        plt.colorbar(c)
        plt.title('left click adds points, right click resets')
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return fig, cid

