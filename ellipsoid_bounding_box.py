import numpy as np


def ellipsoid_bounding_box(mu, Sigma, tau):
    # Ellipse: (x-mu)^T Sigma^-1 (x-mu) < tau^2
    # https://github.com/NickAlger/helper_functions/blob/master/ellipsoid_bounding_box.ipynb
    widths = np.sqrt(Sigma.diagonal()) * tau
    min_pt = mu - widths
    max_pt = mu + widths
    return min_pt, max_pt