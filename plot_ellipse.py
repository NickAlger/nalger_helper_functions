import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_ellipse(mu, Sigma, n_std_tau, ax=None):
    # https://github.com/NickAlger/helper_functions/blob/master/plot_ellipse.ipynb
    if ax is None:
        ax = plt.gca()

    ee, V = np.linalg.eigh(Sigma)
    e_big = ee[1]
    e_small = ee[0]
    v_big = V[:, 1]
    theta = np.arctan(v_big[1] / v_big[0]) * 180. / np.pi

    long_length = n_std_tau * 2. * np.sqrt(e_big)
    short_length = n_std_tau * 2. * np.sqrt(e_small)

    ellipse = Ellipse(mu, width=long_length, height=short_length, angle=theta, facecolor='none', edgecolor='k')
    ax.add_artist(ellipse)
