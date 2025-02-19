import numpy as np
import matplotlib.pyplot as plt

from nalger_helper_functions import mass_matrix_nd

box_widths = (1.2, 3.5)
u_integral_true = -(np.sin(box_widths[1])*(np.cos(box_widths[0]) - 1))

base_grid_shape = (5,3)
grid_shapes = [(int(base_grid_shape[0]*s), int(base_grid_shape[1]*s)) for s in [1, 2, 4, 8, 16, 32, 64, 128]]
hh = []
errs = []
for grid_shape in grid_shapes:
    M = mass_matrix_nd(grid_shape, box_widths)

    xx = np.linspace(0, box_widths[0], grid_shape[0])
    yy = np.linspace(0, box_widths[1], grid_shape[1])
    u = np.outer(np.sin(xx), np.cos(yy)).reshape(-1)

    u_integral = np.sum(M @ u)
    err = np.abs(u_integral - u_integral_true) / np.abs(u_integral_true)
    h = np.max([1.0 / (n-1) for n in grid_shape])
    print('h=', h, ', err=', err)
    hh.append(h)
    errs.append(err)

plt.figure()
plt.loglog(hh, errs)
plt.loglog(hh, np.array(hh)**2, '--')
plt.legend(['error', 'h^2'])
plt.xlabel('h')
plt.ylabel('error')
plt.title('mass matrix error vs mesh size')