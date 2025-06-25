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

RegularGrid = typ.Tuple[
    jnp.ndarray, # min_point = (xmin, ymin)
    jnp.ndarray, # max_point = (xmax, ymax)
    typ.Tuple[int, ...],  # nn = (nx, ny) number of gridpoints per dimension (number of cells+1)
]

def grid_spacings(grid: RegularGrid) -> jnp.ndarray: # hh = (hx, hy)
    min_point, max_point, nn = grid
    hh = (max_point - min_point) / (jnp.array(nn) - 1)
    return hh

TimestepCarry = typ.Tuple[
    jnp.ndarray, # U_cur,  shape=(nx, ny)
    jnp.ndarray, # U_prev, shape=(nx, ny)
    RegularGrid,
    jnp.ndarray, # soundspeed, shape=(nx, ny)
    float, # dt, timestep size
    float, # t_cur, current time
]

source_in_x = jnp.exp(-0.5 * ((X-1.0)**2 + (Y-0.5)**2) / 0.05**2)

def one_timestep(carry: TimestepCarry, unused):
    U_cur, U_prev, grid, soundspeed, dt, t_cur = carry
    hh = grid_spacings(grid)
    source = jnp.sin(40*t_cur) * jnp.exp(-0.5 * t_cur**2 / 0.25**2) * source_in_x

    U_next = 2 * U_cur - U_prev + dt ** 2 * soundspeed ** 2 * laplacian2d(U_cur, hh[0], hh[1]) + source
    t_next = t_cur + dt

    obs_next = U_next
    carry_next = (U_next, U_cur, grid, soundspeed, dt, t_next)

    return carry_next, obs_next


observe_everywhere = lambda U: U

# soundspeed = 1.1 * jnp.ones((nx, ny))
soundspeed = 1.1 + X + Y

dt = 0.5 * np.minimum(hx, hy) / np.max(soundspeed) # CFL condition

# U0 = jnp.exp(-0.5 * ((X-1.0)**2 + (Y-0.5)**2) / 0.05**2)

U0 = jnp.zeros((nx, ny))

grid = (jnp.array([xmin, ymin]), jnp.array([xmax, ymax]), (nx, ny))

carry_init = (U0, U0, grid, soundspeed, dt, 0.0)
U_final, UU = jax.lax.scan(one_timestep, carry_init, None, length=400)

# U_final, UU = jax.lax.scan(scan_func, (U0, U0, hx, hy, soundspeed, dt), None, length=300)

vmax = np.max(UU)
vmin = np.min(UU)
frames = [] # for storing the generated images
fig = plt.figure()
for U in UU:
    frames.append([plt.imshow(U, origin='lower', extent=(xmin, xmax, ymin, ymax), vmin=vmin, vmax=vmax,
                              cmap=cm.Greys_r, animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)
# ani.save('wave1.mp4')
plt.show()

# inhomogeneous soundspeed

if False:
    soundspeed = 1.1 + X + Y

    dt = 0.5 * np.minimum(hx, hy) / np.max(soundspeed) # CFL condition

    U0 = jnp.exp(-0.5 * ((X-1.0)**2 + (Y-0.5)**2) / 0.05**2)

    U_final, UU = jax.lax.scan(scan_func, (U0, U0, hx, hy, soundspeed, dt), None, length=300)

    vmax = np.max(UU)
    vmin = np.min(UU)
    frames = [] # for storing the generated images
    fig = plt.figure()
    for U in UU:
        frames.append([plt.imshow(U, origin='lower', extent=(xmin, xmax, ymin, ymax), vmin=vmin, vmax=vmax,
                                  cmap=cm.Greys_r, animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    # ani.save('wave1.mp4')
    plt.show()





