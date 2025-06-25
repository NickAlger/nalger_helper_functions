import numpy as np
import jax
import jax.numpy as jnp
import typing as typ

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

jax.config.update("jax_enable_x64", True)

# Padding

def padded_shifts(U):
    U_center = jax.lax.pad(U, 0.0, [(1,1,0),(1,1,0)])
    Ux_minus = jax.lax.pad(U, 0.0, [(2,0,0),(1,1,0)])
    Ux_plus = jax.lax.pad(U, 0.0, [(0,2,0),(1,1,0)])
    Uy_minus = jax.lax.pad(U, 0.0, [(1,1,0),(2,0,0)])
    Uy_plus = jax.lax.pad(U, 0.0, [(1,1,0),(0,2,0)])
    return U_center, Ux_minus, Ux_plus, Uy_minus, Uy_plus

nx = 7
ny = 4

U = jnp.ones((nx, ny))
U_center, Ux_minus, Ux_plus, Uy_minus, Uy_plus = padded_shifts(U)

plt.figure()
plt.subplot(2,3,1)
plt.imshow(U_center.T, origin='lower')
plt.title('U')
plt.subplot(2,3,2)
plt.imshow(Ux_minus.T, origin='lower')
plt.title('Ux_minus')
plt.subplot(2,3,3)
plt.imshow(Ux_plus.T, origin='lower')
plt.title('Ux_plus')
plt.subplot(2,3,5)
plt.imshow(Uy_minus.T, origin='lower')
plt.title('Uy_minus')
plt.subplot(2,3,6)
plt.imshow(Uy_plus.T, origin='lower')
plt.title('Uy_plus')

# Laplacian

def laplacian2d_helper(U_center, Ux_minus, Ux_plus, Uy_minus, Uy_plus, hx, hy):
    LU_padded = (Ux_plus - 2 * U_center + Ux_minus) / hx**2 + (Uy_plus - 2 * U_center + Uy_minus) / hy**2
    LU = jax.lax.pad(LU_padded, 0.0, [(-1, -1, 0), (-1, -1, 0)])
    return LU


def laplacian2d(U, hx, hy):
    U_center, Ux_minus, Ux_plus, Uy_minus, Uy_plus = padded_shifts(U)
    return laplacian2d_helper(U_center, Ux_minus, Ux_plus, Uy_minus, Uy_plus, hx, hy)


xmin = 0.0
ymin = 0.0
xmax = 2.0
ymax = 1.0

all_nx = [10, 20, 40, 60, 80, 100, 120, 140, 160]
all_ny = [5, 10, 20, 30, 40, 50, 60, 70, 80]

all_h = []
all_laplacian_relerrs = []
for nx, ny in zip(all_nx, all_ny):
    xx = np.linspace(xmin, xmax, nx)
    yy = np.linspace(ymin, ymax, ny)

    hx = xx[1] - xx[0]
    hy = yy[1] - yy[0]

    X, Y = np.meshgrid(xx, yy, indexing='ij')

    U = np.sin(X) * np.cos(Y)
    LU_true = -2.0 * np.sin(X) * np.cos(Y)

    LU = laplacian2d(U, hx, hy)

    U_interior = jax.lax.pad(U, 0.0, [(-1,-1,0),(-1,-1,0)])
    LU_interior = jax.lax.pad(LU, 0.0, [(-1,-1,0),(-1,-1,0)])
    LU_true_interior = jax.lax.pad(LU_true, 0.0, [(-1,-1,0),(-1,-1,0)])

    h = np.sqrt(hx**2 + hy**2)
    laplacian_relerr = np.linalg.norm(LU_true_interior - LU_interior) / np.linalg.norm(LU_true_interior)
    print('h=', h, ', laplacian_relerr=', laplacian_relerr)

    all_h.append(h)
    all_laplacian_relerrs.append(laplacian_relerr)

plt.figure()
plt.loglog(all_h, all_laplacian_relerrs)
plt.xlabel('h')
plt.ylabel('relerr')
plt.title('Laplacian finite difference error')


plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(U_interior.T, origin='lower', extent=(xmin+hx, xmax-hx, ymin+hy, ymax-hy))
plt.title('U_interior')
plt.subplot(1,3,2)
plt.imshow(LU_interior.T, origin='lower', extent=(xmin+hx, xmax-hx, ymin+hy, ymax-hy))
plt.title('LU_interior')
plt.subplot(1,3,3)
plt.imshow(LU_true_interior.T, origin='lower', extent=(xmin+hx, xmax-hx, ymin+hy, ymax-hy))
plt.title('LU_true_interior')

# Wave timestepping

def one_timestep(carry, unused):
    U1, U0, min_point, max_point, soundspeed, dt, ii, source_in_x, source_in_t = carry
    nx, ny = U1.shape
    hh = (max_point - min_point) / (jnp.array([nx, ny]) - 1)
    hx = hh[0]
    hy = hh[1]

    source = source_in_t[ii] * source_in_x

    LU1_int = ((U1[2:,1:-1] - 2 * U1[1:-1,1:-1] + U1[:-2,1:-1]) / hx**2 +
               (U1[1:-1,2:] - 2 * U1[1:-1,1:-1] + U1[1:-1,:-2]) / hy**2)

    U2_int = 2 * U1[1:-1,1:-1] - U0[1:-1,1:-1] + dt ** 2 * soundspeed[1:-1,1:-1] ** 2 * LU1_int + source[1:-1,1:-1]

    U2_top = jnp.zeros((nx,1)) # Dirichlet zero top
    # U2_top = U1[:,-2].reshape((nx,1)) # Neumann zero top (maybe bad? timestep lag)


    U2_bot = (U1[:,0] + soundspeed[:,0] * (dt / hy) * (U1[:,1] - U1[:,0])).reshape((nx,1)) # c u_y = u_t ABC

    U2_left = (U1[0,1:-1] + soundspeed[0,1:-1] * (dt / hx) * (U1[1,1:-1] - U1[0,1:-1])).reshape((1, ny-2)) # c u_x = u_t ABC
    U2_right = (U1[-1, 1:-1] + soundspeed[-1, 1:-1] * (dt / hx) * (U1[-2, 1:-1] - U1[-1, 1:-1])).reshape((1, ny-2)) # c u_x = u_t ABC

    U2 = jnp.hstack([U2_bot, jnp.vstack([U2_left, U2_int, U2_right]), U2_top])

    obs_next = U2
    carry_next = (U2, U1, min_point, max_point, soundspeed, dt, ii+1, source_in_x, source_in_t)

    return carry_next, obs_next


def cfl_max_timestep(min_point, max_point, soundspeed):
    hh = (max_point - min_point) / (jnp.array(soundspeed.shape) - 1)
    dt_max = np.min(hh) / np.max(soundspeed)  # CFL condition
    return dt_max


observe_everywhere = lambda U: U

xmin = 0.0
ymin = 0.0
xmax = 2.0
ymax = 1.0

min_point = jnp.array([xmin, ymin])
max_point = jnp.array([xmax, ymax])

grid_shape = (160, 80)

xx = np.linspace(xmin, xmax, grid_shape[0])
yy = np.linspace(ymin, ymax, grid_shape[1])
X, Y = np.meshgrid(xx, yy, indexing='ij')


soundspeed = 1.1 + X + Y

plt.figure()
plt.imshow(soundspeed.T, origin='lower', extent=(xmin, xmax, ymin, ymax))
plt.title('soundspeed')

t_final = 1.0

dt0 = 0.5 * cfl_max_timestep(min_point, max_point, soundspeed)
num_timesteps = int(np.ceil(t_final / dt0))
tt = jnp.linspace(0.0, t_final, num_timesteps + 1)

dt = t_final / num_timesteps

source_in_x = jnp.exp(-0.5 * ((X-1.0)**2 + (Y-0.5)**2) / 0.025**2)
source_in_t = jnp.sin(60*tt) * jnp.exp(-0.5 * tt**2 / 0.075**2)

plt.figure()
plt.imshow(source_in_x.T, origin='lower', extent=(xmin, xmax, ymin, ymax))
plt.title('source_in_x')

plt.figure()
plt.plot(tt, source_in_t)
plt.title('source_in_t')

U0 = jnp.zeros((nx, ny))

carry_init = (U0, U0, min_point, max_point, soundspeed, dt, 0, source_in_x, source_in_t)
U_final, UU = jax.lax.scan(one_timestep, carry_init, None, length=num_timesteps)


vmax = np.max(UU)
vmin = np.min(UU)
frames = [] # for storing the generated images
fig = plt.figure()
for U in UU:
    frames.append([plt.imshow(U.T, origin='lower', extent=(xmin, xmax, ymin, ymax), vmin=vmin, vmax=vmax,
                              cmap=cm.Greys_r, animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1)
# ani.save('wave1.mp4')
plt.show()





