import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

from nalger_helper_functions.laplacian_on_box import *

#### Mix of Dirichlet and Neumann B.C.'s ####

grid_shape = (93, 68)
box_widths = (2*np.pi, 3*np.pi/2)
left_bcs = ['Dirichlet', 'Neumann']
right_bcs = ['Dirichlet', 'Dirichlet']

L = laplacian_nd(grid_shape, box_widths, left_bcs, right_bcs)

# Solve with random function

N = np.prod(grid_shape)
x = np.random.randn(N)
y = spla.spsolve(L, x)

plt.figure(figsize=(12,6))
plt.imshow(y.reshape(grid_shape).T, extent=[0, box_widths[0], 0, box_widths[1]], origin='lower')
plt.title('inverse Laplacian times randn\nx BCs: [D, D]\ny BCs: [N, D]')
plt.colorbar()

# Solve with sin(x)cos(y)

xx = np.linspace(0, box_widths[0], grid_shape[0])
yy = np.linspace(0, box_widths[1], grid_shape[1])
u = np.outer(np.sin(xx), np.cos(yy)).reshape(-1)
v = spla.spsolve(L, u)

# v = -L^-1 u = u / 2 since
# L u = -(d^2/dx^2 + d^2/dy^2) sin(x)cos(y) = sin(x)cos(x) + sin(x)cos(x) = 2 sin(x)cos(x) = 2 u
max_h = np.max(1.0 / (np.array(grid_shape)-1))
err = np.linalg.norm(v - u/2) / np.linalg.norm(u)
print('max_h=', max_h, ', err=', err)

plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.imshow(u.reshape(grid_shape).T, extent=[0, box_widths[0], 0, box_widths[1]], origin='lower')
plt.title('sin(x) * cos(y)')
plt.colorbar()

plt.subplot(2,1,2)
plt.imshow(v.reshape(grid_shape).T, extent=[0, box_widths[0], 0, box_widths[1]], origin='lower')
plt.title('inverse Laplacian times sin(x) * cos(y)\nx BCs: [D, D]\ny BCs: [N, D]')
plt.colorbar()

# Convergence test:

nx0 = 9
ny0 = 6
grid_shapes = [(int(nx0*s), int(ny0*s)) for s in np.logspace(np.log10(1), np.log10(20), 10)]
hh = []
errs = []
for grid_shape in grid_shapes:
    L = laplacian_nd(grid_shape, box_widths, left_bcs, right_bcs)

    xx = np.linspace(0, box_widths[0], grid_shape[0])
    yy = np.linspace(0, box_widths[1], grid_shape[1])
    u = np.outer(np.sin(xx), np.cos(yy)).reshape(-1)
    v = spla.spsolve(L, u)

    # v = -L^-1 u = u / 2 since
    # L u = -(d^2/dx^2 + d^2/dy^2) sin(x)cos(y) = sin(x)cos(x) + sin(x)cos(x) = 2 sin(x)cos(x) = 2 u
    max_h = np.max(1.0 / (np.array(grid_shape) - 1))
    err = np.linalg.norm(v - u / 2) / np.linalg.norm(u)
    print('max_h=', max_h, ', err=', err)
    hh.append(max_h)
    errs.append(err)

plt.figure()
plt.loglog(hh, errs)
plt.loglog(hh, hh, '--')
plt.legend(['error', 'h'])
plt.xlabel('h')
plt.ylabel('error')
plt.title('Laplacian error vs. mesh size\nx BCs: [D, D]\ny BCs: [N, D]')

# Periodic B.C.'s


#### Periodic B.C.'s ####

grid_shape = (93, 88)
box_widths = (2*np.pi, 2*np.pi)

L = periodic_laplacian_nd(grid_shape, box_widths)

# Apply to sin(x-1)cos(y-1)

xx = np.linspace(0, box_widths[0], grid_shape[0], endpoint=False)
yy = np.linspace(0, box_widths[1], grid_shape[1], endpoint=False)
u = np.outer(np.sin(xx - 1.0), np.cos(yy - 1.0)).reshape(-1)
v = L @ u

# v = L u = 2 u
# L u = -(d^2/dx^2 + d^2/dy^2) sin(x)cos(y) = sin(x)cos(x) + sin(x)cos(x) = 2 sin(x)cos(x) = 2 u
max_h = np.max(1.0 / (np.array(grid_shape)-1))
err = np.linalg.norm(v - 2*u) / np.linalg.norm(u)
print('max_h=', max_h, ', err=', err)

plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.imshow(u.reshape(grid_shape).T, extent=[0, box_widths[0], 0, box_widths[1]], origin='lower')
plt.title('sin(x-1) * cos(y-1)')
plt.colorbar()

plt.subplot(2,1,2)
plt.imshow(v.reshape(grid_shape).T, extent=[0, box_widths[0], 0, box_widths[1]], origin='lower')
plt.title('Laplacian times sin(x-1) * cos(y-1)\nPeriodic BCs')
plt.colorbar()

# Convergence test:

nx0 = 9
ny0 = 6
grid_shapes = [(int(nx0*s), int(ny0*s)) for s in np.logspace(np.log10(1), np.log10(20), 10)]
hh = []
errs = []
for grid_shape in grid_shapes:
    L = periodic_laplacian_nd(grid_shape, box_widths)

    xx = np.linspace(0, box_widths[0], grid_shape[0], endpoint=False)
    yy = np.linspace(0, box_widths[1], grid_shape[1], endpoint=False)
    u = np.outer(np.sin(xx-1), np.cos(yy-1)).reshape(-1)
    v = L @ u

    # v = -L^-1 u = u / 2 since
    # L u = -(d^2/dx^2 + d^2/dy^2) sin(x)cos(y) = sin(x)cos(x) + sin(x)cos(x) = 2 sin(x)cos(x) = 2 u
    max_h = np.max(1.0 / (np.array(grid_shape) - 1))
    err = np.linalg.norm(v - 2*u) / np.linalg.norm(u)
    print('max_h=', max_h, ', err=', err)
    hh.append(max_h)
    errs.append(err)

plt.figure()
plt.loglog(hh, errs)
plt.loglog(hh, hh, '--')
plt.legend(['error', 'h'])
plt.xlabel('h')
plt.ylabel('error')
plt.title('Laplacian error vs. mesh size\nPeriodic BCs')


