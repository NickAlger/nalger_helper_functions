import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_rectangle(minpt, maxpt, ax=None, edgecolor='k', linewidth=1, facecolor='none'):
    # https://github.com/NickAlger/helper_functions/blob/master/plot_rectangle.ipynb
    if ax is None:
        ax = plt.gca()

    delta = maxpt - minpt
    rect = patches.Rectangle(minpt, delta[0], delta[1],
                             linewidth=linewidth,
                             edgecolor=edgecolor,
                             facecolor=facecolor)
    ax.add_patch(rect)