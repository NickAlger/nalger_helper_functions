import numpy as np
from fenics import *
from matplotlib.image import imread
from scipy.interpolate import interpn


def load_image_into_fenics(function_space_V, image_filename):
    img = imread(image_filename)
    gray_img0 = np.mean(img, axis=-1)[::-1, :]
    # [::-1,:] flip because image array goes top left to bottom right whereas we want bottom left to top right

    cmin = np.min(gray_img0)
    cmax = np.max(gray_img0)
    gray_img = (gray_img0 - cmin) / (cmax - cmin)

    coords = function_space_V.tabulate_dof_coordinates()
    xmin = np.min(coords[:, 0])
    xmax = np.max(coords[:, 0])
    ymin = np.min(coords[:, 1])
    ymax = np.max(coords[:, 1])

    aspect_ratio = gray_img.shape[0] / gray_img.shape[1]
    xmax2 = (ymax - ymin) / aspect_ratio

    xx = np.linspace(xmin, xmax2, gray_img.shape[1])
    yy = np.linspace(ymin, ymax, gray_img.shape[0])

    f = Function(function_space_V)
    f.vector()[:] = interpn((xx, yy), gray_img.swapaxes(0, 1), coords)

    return f