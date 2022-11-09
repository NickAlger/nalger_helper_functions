import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.fft as fft
from scipy.spatial import KDTree
from tqdm.auto import tqdm
from collections.abc import Callable


# --------    Helpers    --------

def csr_fenics2scipy(A_fenics):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/blob/master/jupyter_notebooks/csr_conversions.ipynb
    ai, aj, av = dl.as_backend_type(A_fenics).mat().getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai))
    return A_scipy

def plot_ellipse(mu, Sigma, n_std_tau, ax=None, **kwargs):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/plot_ellipse.ipynb
    if ax is None:
        ax = plt.gca()

    ee, V = np.linalg.eigh(Sigma)
    e_big = ee[1]
    e_small = ee[0]
    v_big = V[:, 1]
    theta = np.arctan(v_big[1] / v_big[0]) * 180. / np.pi

    long_length = n_std_tau * 2. * np.sqrt(e_big)
    short_length = n_std_tau * 2. * np.sqrt(e_small)

    if not ('facecolor' in kwargs):
        kwargs['facecolor'] = 'none'

    if not ('edgecolor' in kwargs):
        kwargs['edgecolor'] = 'k'

    ellipse = Ellipse(mu, width=long_length, height=short_length, angle=theta, **kwargs)
    ax.add_artist(ellipse)


# --------    Window2D: class for evaluating a function on a rectangular grid    --------

class Window2D:
    def __init__(me, nx=20, ny=20):
        me.nx = nx
        me.ny = ny
        me.num_cells_x = nx-1
        me.num_cells_y = ny-1
        me.d = 2

        me.mesh = dl.UnitSquareMesh(me.num_cells_x, me.num_cells_y)

        me.mesh.coordinates()[:] = 2.0*me.mesh.coordinates() - 1.0
        me.mesh_coords0 = me.mesh_coords()

        me.bot_left_ind = np.argwhere(np.logical_and(np.abs(me.mesh_coords0[:,0] + 1.0) < 1e-14,
                                                     np.abs(me.mesh_coords0[:,1] + 1.0) < 1e-14))[0][0]

        me.bot_right_ind = np.argwhere(np.logical_and(np.abs(me.mesh_coords0[:,0] - 1.0) < 1e-14,
                                                      np.abs(me.mesh_coords0[:,1] + 1.0) < 1e-14))[0][0]

        me.top_right_ind = np.argwhere(np.logical_and(np.abs(me.mesh_coords0[:,0] - 1.0) < 1e-14,
                                                      np.abs(me.mesh_coords0[:,1] - 1.0) < 1e-14))[0][0]

        me.top_left_ind = np.argwhere(np.logical_and(np.abs(me.mesh_coords0[:,0] + 1.0) < 1e-14,
                                                     np.abs(me.mesh_coords0[:,1] - 1.0) < 1e-14))[0][0]

        me.Vh = dl.FunctionSpace(me.mesh, 'CG', 1)

        me.dof_coords0 = me.Vh.tabulate_dof_coordinates().copy()
        me.dof_sort_inds = np.lexsort(me.dof_coords0.T)

        me.vol = 1.0
        me.mu = np.array([0.0, 0.0])
        me.Sigma = np.eye(2)
        me.tau = 3.0

        me.ee, me.P = sla.eigh(me.Sigma)

    def mesh_coords(me):
        return me.mesh.coordinates().copy()

    def transform(me, vol, mu, Sigma, tau):
        if vol is not None:
            me.vol = vol
        if mu is not None:
            me.mu = np.array(mu).reshape(2)
        if Sigma is not None:
            me.Sigma = np.array(Sigma).reshape((2,2))
        if tau is not None:
            me.tau = tau

        me.ee, me.P = sla.eigh(me.Sigma)
        scaled_coords = me.mesh_coords0 * np.sqrt(me.ee).reshape((1,2)) * me.tau
        me.mesh.coordinates()[:] = np.dot(scaled_coords, me.P.T) + me.mu.reshape((1,2))

    def plot_window(me):
        cc = me.mesh_coords()
        bot_inds = [me.bot_left_ind, me.bot_right_ind]
        left_inds = [me.bot_left_ind, me.top_left_ind]
        top_inds = [me.top_left_ind, me.top_right_ind]
        right_inds = [me.top_right_ind, me.bot_right_ind]
        plt.plot(cc[bot_inds, 0], cc[bot_inds, 1], 'r', linewidth=1)
        plt.plot(cc[left_inds, 0], cc[left_inds, 1], 'b', linewidth=1)
        plt.plot(cc[top_inds, 0], cc[top_inds, 1], 'k', linewidth=1)
        plt.plot(cc[right_inds, 0], cc[right_inds, 1], 'k', linewidth=1)
        plot_ellipse(me.mu,me.Sigma,me.tau)
        # plt.plot(cc[:, 0], cc[:, 1], '.k', markersize=1)

    def get_windowed_function(me, u: dl.Function) -> np.ndarray:
        u.set_allow_extrapolation(True)
        u_window = dl.interpolate(u, me.Vh)
        U = u_window.vector()[me.dof_sort_inds].reshape((me.nx, me.ny)) / me.vol
        return U


# --------    Build dense spatially varying impulse response (SVIR)    --------

def get_SVIR(apply_A, solve_M, Vh, all_vol, all_mu, all_Sigma, tau, nx=20, ny=20):
    SVIR = np.zeros((Vh.dim(), nx, ny))
    win = Window2D(nx, ny)
    print('building SVIR')
    for kk in tqdm(range(Vh.dim())):
        ek = np.zeros(Vh.dim())
        ek[kk] = 1.0
        psi_k = dl.Function(Vh)
        psi_k.vector()[:] = solve_M(apply_A(solve_M(ek)))
        win.transform(all_vol[kk], all_mu[kk], all_Sigma[kk], tau)
        SVIR[kk,:,:] = win.get_windowed_function(psi_k)
    print('done building SVIR')
    return SVIR


def SVIR_to_symbol(SVIR):
    symbol = np.zeros(SVIR.shape, dtype=complex)
    print('building symbol from SVIR')
    for kk in tqdm(range(SVIR.shape[0])):
        symbol[kk,:,:] = fft.fftshift(fft.fftn(fft.ifftshift(SVIR[kk,:,:])))
    print('done building symbol from SVIR')
    return symbol


# --------    SVIR / symbol plotting    --------

class InteractiveImpulsePlots:
    def __init__(me,
                 apply_A:     Callable[[np.ndarray], np.ndarray], # R^N -> R^N
                 solve_M:     Callable[[np.ndarray], np.ndarray],  # R^N -> R^N
                 all_scaling: np.ndarray, # shape=(N,)
                 all_mu:      np.ndarray, # shape=(N,d)
                 all_Sigma:   np.ndarray, # shape=(N,d,d)
                 tau: float,
                 Vh: dl.FunctionSpace, # Vh.dim()=N
                 nx : int = 20,
                 ny : int = 20,
                 SVIR: np.ndarray = None,  # shape=(N,m,m)
                 symbol: np.ndarray = None # shape=(N,m,m)
                 ):
        me.apply_A = apply_A
        me.solve_M = solve_M
        me.all_scaling = all_scaling
        me.all_mu = all_mu
        me.all_Sigma = all_Sigma
        me.tau = tau
        me.Vh = Vh
        me.nx = nx
        me.ny = ny

        me.psi_k = dl.Function(me.Vh)

        if SVIR is None:
            me.SVIR = get_SVIR(me.apply_A, me.solve_M, me.Vh, me.all_scaling, me.all_mu, me.all_Sigma, me.tau, me.nx, me.ny)
        else:
            me.SVIR = SVIR

        if symbol is None:
            me.symbol = SVIR_to_symbol(me.SVIR)
        else:
            me.symbol = symbol

        me.dof_coords = me.Vh.tabulate_dof_coordinates()
        me.kdtree = KDTree(me.dof_coords)

        me.xlims = (np.min(me.dof_coords[:,0]), np.max(me.dof_coords[:,0]))
        me.ylims = (np.min(me.dof_coords[:,1]), np.max(me.dof_coords[:,1]))

        me.fig = plt.figure(figsize=(10, 10))
        me.reset_plots()

        me.win = Window2D(nx=me.SVIR.shape[1], ny=me.SVIR.shape[2])

        me.cid = me.fig.canvas.mpl_connect('button_press_event', me.onclick)
        plt.show()

    def onclick(me, event):
        if event.inaxes == me.ax_u:
            x = event.xdata
            y = event.ydata
            k = me.kdtree.query([x, y])[1]
            x_nearest = me.dof_coords[k, 0]
            y_nearest = me.dof_coords[k, 1]

            print('clicked point=(', x, ',', y, ')')
            print('nearest dof=(', x_nearest, ',', y_nearest, ')')

            ek = np.zeros(me.Vh.dim())
            ek[k] = 1.0
            me.psi_k.vector()[:] = me.solve_M(me.apply_A(me.solve_M(ek)))

            scaling_k = me.all_scaling[k]
            mu_k = me.all_mu[k,:]
            Sigma_k = me.all_Sigma[k,:,:]
            me.win.transform(scaling_k, mu_k, Sigma_k, me.tau)

            me.reset_plots()

            plt.sca(me.ax_u)

            plt.sca(me.ax_u)
            me.ax_u.plot(dof_coords[k, 0], dof_coords[k, 1], '.r')
            me.win.plot_window()

            plt.sca(me.ax_psi)
            im_impulse = me.ax_psi.imshow(me.SVIR[k,:,:], interpolation='bilinear', origin='lower')
            divider = make_axes_locatable(me.ax_psi)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            me.fig.colorbar(im_impulse, cax=cax, orientation='vertical')

            plt.sca(me.ax_symbol_real)
            im_impulse = me.ax_symbol_real.imshow(me.symbol[k, :, :].real, interpolation='bilinear', origin='lower')
            divider = make_axes_locatable(me.ax_symbol_real)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            me.fig.colorbar(im_impulse, cax=cax, orientation='vertical')

            plt.sca(me.ax_symbol_imag)
            im_impulse = me.ax_symbol_imag.imshow(me.symbol[k, :, :].imag, interpolation='bilinear', origin='lower')
            divider = make_axes_locatable(me.ax_symbol_imag)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            me.fig.colorbar(im_impulse, cax=cax, orientation='vertical')

            plt.draw()

    def reset_plots(me):
        me.fig.clear()

        me.ax_u = me.fig.add_subplot(2, 2, 1)
        me.ax_psi = me.fig.add_subplot(2, 2, 2)
        me.ax_symbol_real = me.fig.add_subplot(2, 2, 3)
        me.ax_symbol_imag = me.fig.add_subplot(2, 2, 4)

        plt.sca(me.ax_u)
        im_u = dl.plot(me.psi_k)
        divider = make_axes_locatable(me.ax_u)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        me.fig.colorbar(im_u, cax=cax, orientation='vertical')
        me.ax_u.set_xlim(*me.xlims)
        me.ax_u.set_ylim(*me.ylims)

        me.ax_u.set_title('Click to plot impulse')
        me.ax_psi.set_title(r'$\phi_x$')
        me.ax_symbol_real.set_title(r'$\Re\left(\widehat{\phi}_x\right)$')
        me.ax_symbol_imag.set_title(r'$\Im\left(\widehat{\phi}_x\right)$')


class SliceViewer:
    def __init__(me,
                 T: np.ndarray, # shape=(N,m1,m2)
                 Vh: dl.FunctionSpace # Vh.dim() = N
                 ):
        me.T = T
        me.Vh = Vh

        me.N, me.m1, me.m2 = T.shape

        me.weights_real = dl.Function(me.Vh)
        me.weights_imag = dl.Function(me.Vh)

        me.fig = plt.figure(figsize=(10, 10))
        me.reset_plots()

        me.cid = me.fig.canvas.mpl_connect('button_press_event', me.onclick)
        plt.show()

    def onclick(me, event):
        if event.inaxes == me.ax_clicker:
            x = event.xdata
            y = event.ydata

            ii = int(np.min([me.m1-1, np.max([0, np.round(x)])]))
            jj = int(np.min([me.m2-1, np.max([0, np.round(y)])]))

            print('clicked point=(', x, ',', y, ')')
            print('nearest inds=(', ii, ',', jj, ')')

            ww_real = me.T[:, ii, jj].real.copy()
            ww_imag = me.T[:, ii, jj].imag.copy()

            ww_real[np.isnan(ww_real)] = 0.0
            ww_imag[np.isnan(ww_imag)] = 0.0

            me.weights_real.vector()[:] = ww_real
            me.weights_imag.vector()[:] = ww_imag

            me.reset_plots()

            plt.sca(me.ax_clicker)
            me.ax_clicker.plot(ii, jj, '.r')

            plt.sca(me.ax_weights_real)
            cm_weights_real = dl.plot(me.weights_real)
            divider = make_axes_locatable(me.ax_weights_real)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            me.fig.colorbar(cm_weights_real, cax=cax, orientation='vertical')

            plt.sca(me.ax_weights_imag)
            cm_weights_imag = dl.plot(me.weights_imag)
            divider = make_axes_locatable(me.ax_weights_imag)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            me.fig.colorbar(cm_weights_imag, cax=cax, orientation='vertical')

            plt.draw()

    def reset_plots(me):
        me.fig.clear()

        me.ax_weights_real = me.fig.add_subplot(2, 2, 1)
        me.ax_weights_imag = me.fig.add_subplot(2, 2, 2)
        me.ax_clicker = me.fig.add_subplot(2, 2, 3)

        plt.sca(me.ax_clicker)
        im_clicker = me.ax_clicker.imshow(0.0*me.T[0, :, :].real, interpolation='bilinear', origin='lower')
        divider = make_axes_locatable(me.ax_clicker)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        me.fig.colorbar(im_clicker, cax=cax, orientation='vertical')

        me.ax_clicker.set_xlim([0, me.m1-1])
        me.ax_clicker.set_ylim([0, me.m2-1])

        me.ax_clicker.set_title(r'Click (i,j) point to plot $w_{ij}$')
        me.ax_weights_real.set_title(r'$\Re\left(w_{ij}(x)\right)$')
        me.ax_weights_imag.set_title(r'$\Im\left(w_{ij}(x)\right)$')



#

import meshio

vertices = np.loadtxt('mesh_vertices.txt')
cells = np.loadtxt('mesh_cells.txt', dtype=int)

mesh_meshio = meshio.Mesh(vertices, [("triangle", cells)])
mesh_meshio.write("mesh.xml")
mesh = dl.Mesh("mesh.xml")

Hdgn = np.loadtxt('Hdgn_array.txt')

Vh = dl.FunctionSpace(mesh, 'CG', 1)
M = csr_fenics2scipy(dl.assemble(dl.TestFunction(Vh)*dl.TrialFunction(Vh)*dl.dx)) # mass matrix
mass_lumps = M * np.ones(M.shape[1])

apply_Hdgn = lambda x: np.dot(Hdgn, x)
solve_ML = lambda x: x / mass_lumps

dof_coords = Vh.tabulate_dof_coordinates().copy()

all_vol = solve_ML(apply_Hdgn(np.ones(Vh.dim())))

all_mu = np.zeros((Vh.dim(), 2))
all_mu[:,0] = solve_ML(apply_Hdgn(dof_coords[:,0])) / all_vol
all_mu[:,1] = solve_ML(apply_Hdgn(dof_coords[:,1])) / all_vol

all_Sigma = np.zeros((Vh.dim(), 2, 2))
all_Sigma[:,0,0] = solve_ML(apply_Hdgn(dof_coords[:,0]*dof_coords[:,0])) / all_vol - all_mu[:,0]*all_mu[:,0]
all_Sigma[:,0,1] = solve_ML(apply_Hdgn(dof_coords[:,0]*dof_coords[:,1])) / all_vol - all_mu[:,0]*all_mu[:,1]
all_Sigma[:,1,1] = solve_ML(apply_Hdgn(dof_coords[:,1]*dof_coords[:,1])) / all_vol - all_mu[:,1]*all_mu[:,1]
all_Sigma[:,1,0] = all_Sigma[:,0,1]

# Ellipsoid = {x: (x-mu)^T Sigma^-1 (x-mu) < tau^2}

all_sqrt_det_Sigma = np.sqrt(np.array([np.linalg.det(all_Sigma[kk,:,:]) for kk in range(all_Sigma.shape[0])]))

for kk in range(all_Sigma.shape[0]):
    ee, P = sla.eigh(all_Sigma[kk,:,:])
    if np.any(ee <= 0):
        print('Bad ellipsoid. kk=', kk, ', ee=', ee)
        all_Sigma[kk, :, :] = np.dot(P, np.dot(np.diag(np.abs(ee)), P.T))

tau = 4.0

# IRP = InteractiveImpulsePlots(apply_Hdgn, solve_ML, all_vol, all_mu, all_Sigma, tau, Vh)
# IRP = InteractiveImpulsePlots(apply_Hdgn, solve_ML, all_vol/all_sqrt_det_Sigma, all_mu, all_Sigma, tau, Vh)
IRP = InteractiveImpulsePlots(apply_Hdgn, solve_ML, np.ones(all_vol.shape), all_mu, all_Sigma, tau, Vh)

# SV1 = SliceViewer(IRP.SVIR, IRP.Vh)
SV2 = SliceViewer(IRP.symbol, IRP.Vh)

SVIR = IRP.SVIR
SVIR[np.isnan(SVIR)] = 0.0
SVIR[np.isinf(SVIR)] = 0.0

U, ss, Vt = sla.svd(SVIR.reshape((Vh.dim(), -1)), 0)

VVt = Vt.reshape((-1, IRP.SVIR.shape[1], IRP.SVIR.shape[2]))

plt.figure()
plt.semilogy(ss)
plt.title('SVIR singular values')

for k in range(5):
    plt.figure()

    plt.subplot(1,2,1)
    plt.imshow(VVt[k,:,:])
    plt.title('SVIR SVD, k=' + str(k))

    plt.subplot(1,2,2)
    uk = dl.Function(Vh)
    uk.vector()[:] = U[:,k]
    cm = dl.plot(uk)
    plt.colorbar(cm)