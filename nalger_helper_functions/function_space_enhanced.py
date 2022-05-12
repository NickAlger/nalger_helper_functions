import numpy as np
import dolfin as dl
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree
from nalger_helper_functions import csr_fenics2scipy


class FunctionSpaceEnhanced(dl.FunctionSpace):
    def __init__(me, *args, **kwargs):
        dl.FunctionSpace.__init__(me, *args, **kwargs)

        me._coords = None
        me._mass_matrix_petsc = None
        me._mass_matrix_scipy = None
        me._mass_matrix_LUSolver = None
        me._kd_tree = None
        me._mass_lumps_rowsum_petsc = None
        me._mass_lumps_rowsum_numpy = None
        me._mass_lumps_diagonal_petsc = None
        me._mass_lumps_diagonal_numpy = None

        # me.num_applies = 0

        me.apply_mass_matrix_linop = spla.LinearOperator((me.dim(), me.dim()),
                                                         matvec=me.apply_mass_matrix,
                                                         rmatvec=me.apply_mass_matrix)

        me.solve_mass_matrix_LU_linop = spla.LinearOperator((me.dim(), me.dim()),
                                                            matvec=me.solve_mass_matrix_LU,
                                                            rmatvec=me.solve_mass_matrix_LU)

        me.apply_lumped_mass_matrix_diagonal_linop = spla.LinearOperator((me.dim(), me.dim()),
                                                                         matvec=me.apply_lumped_mass_matrix_diagonal,
                                                                         rmatvec=me.apply_lumped_mass_matrix_diagonal)

        me.apply_lumped_mass_matrix_rowsum_linop = spla.LinearOperator((me.dim(), me.dim()),
                                                                       matvec=me.apply_lumped_mass_matrix_rowsum,
                                                                       rmatvec=me.apply_lumped_mass_matrix_rowsum)

        me.solve_lumped_mass_matrix_diagonal_linop = spla.LinearOperator((me.dim(), me.dim()),
                                                                         matvec=me.solve_lumped_mass_matrix_diagonal,
                                                                         rmatvec=me.solve_lumped_mass_matrix_diagonal)

        me.solve_lumped_mass_matrix_rowsum_linop = spla.LinearOperator((me.dim(), me.dim()),
                                                                       matvec=me.solve_lumped_mass_matrix_rowsum,
                                                                       rmatvec=me.solve_lumped_mass_matrix_rowsum)

        me.solve_mass_matrix_PCG_linop = spla.LinearOperator((me.dim(), me.dim()),
                                                             matvec=me.solve_mass_matrix_PCG,
                                                             rmatvec=me.solve_mass_matrix_PCG)

        me.apply_M = me.apply_mass_matrix
        me.apply_MLd = me.apply_lumped_mass_matrix_diagonal
        me.apply_MLr = me.apply_lumped_mass_matrix_rowsum
        me.solve_M_LU = me.solve_mass_matrix_LU
        me.solve_MLd = me.solve_lumped_mass_matrix_diagonal
        me.solve_MLr = me.solve_lumped_mass_matrix_rowsum

        me.apply_M_linop = me.apply_mass_matrix_linop
        me.apply_MLd_linop = me.apply_lumped_mass_matrix_diagonal_linop
        me.apply_MLr_linop = me.apply_lumped_mass_matrix_rowsum_linop
        me.solve_M_LU_linop = me.solve_mass_matrix_LU_linop
        me.solve_MLd_linop = me.solve_lumped_mass_matrix_diagonal_linop
        me.solve_MLr_linop = me.solve_lumped_mass_matrix_rowsum_linop

    @property
    def coords(me):
        if me._coords is None:
            me._coords = me.tabulate_dof_coordinates()
        return me._coords

    @property
    def mass_matrix_petsc(me):
        if me._mass_matrix_petsc is None:
            me._mass_matrix_petsc = dl.assemble(me.test_function * me.trial_function * dl.dx)
        return me._mass_matrix_petsc

    @property
    def mass_matrix_scipy(me):
        if me._mass_matrix_scipy is None:
            me._mass_matrix_scipy = csr_fenics2scipy(me.mass_matrix_petsc)
        return me._mass_matrix_scipy

    def apply_mass_matrix(me, x):
        # print('me.num_applies=', me.num_applies)
        # me.num_applies += 1
        if isinstance(x, np.ndarray):
            Mx = me.petsc2numpy(me.mass_matrix_petsc * me.numpy2petsc(x))
        else:
            Mx = me.mass_matrix_petsc * x
        return Mx

    def get_vector(me):
        x_petsc = dl.Vector()
        me.mass_matrix_petsc.init_vector(x_petsc,1)
        return x_petsc

    def get_function(me):
        return dl.Function(me)

    @property
    def test_function(me):
        return dl.TestFunction(me)

    @property
    def trial_function(me):
        return dl.TrialFunction(me)

    def numpy2petsc(me, x_numpy):
        x_petsc = me.get_vector()
        x_petsc[:] = x_numpy
        return x_petsc

    def petsc2numpy(me, x_petsc):
        return x_petsc[:]

    @property
    def kd_tree(me):
        if me._kd_tree is None:
            me._kd_tree = KDTree(me.coords)
        return me._kdtree

    @property
    def mass_matrix_LUSolver(me):
        if me._mass_matrix_LUSolver is None:
            me._mass_matrix_LUSolver = dl.LUSolver(me.mass_matrix_petsc)
        return me._mass_matrix_LUSolver

    def solve_mass_matrix_LU(me, y):
        x_petsc = me.get_vector()
        if isinstance(y, np.ndarray):
            me.mass_matrix_LUSolver.solve(x_petsc, me.numpy2petsc(y))
            x = me.petsc2numpy(x_petsc)
        else:
            me.mass_matrix_LUSolver.solve(x_petsc, y)
            x = x_petsc
        return x

    @property
    def mass_lumps_rowsum_petsc(me):
        if me._mass_lumps_rowsum_petsc is None:
            me._mass_lumps_rowsum_petsc = dl.assemble(dl.Constant(1.0) * me.test_function * dl.dx)
        return me._mass_lumps_rowsum_petsc

    @property
    def mass_lumps_diagonal_petsc(me):
        if me._mass_lumps_diagonal_petsc is None:
            me._mass_lumps_diagonal_petsc = me.get_vector()
            me.mass_matrix_petsc.get_diagonal(me._mass_lumps_diagonal_petsc)
        return me._mass_lumps_diagonal_petsc

    @property
    def mass_lumps_rowsum_numpy(me):
        if me._mass_lumps_rowsum_numpy is None:
            me._mass_lumps_rowsum_numpy = me.petsc2numpy(me.mass_lumps_rowsum_petsc)
        return me._mass_lumps_rowsum_numpy

    @property
    def mass_lumps_diagonal_numpy(me):
        if me._mass_lumps_diagonal_numpy is None:
            me._mass_lumps_diagonal_numpy = me.petsc2numpy(me.mass_lumps_diagonal_petsc)
        return me._mass_lumps_diagonal_numpy

    def apply_lumped_mass_matrix_diagonal(me, x):
        if isinstance(x, np.ndarray):
            MLx = me.mass_lumps_diagonal_numpy * x
        else:
            MLx = me.mass_lumps_diagonal_petsc * x
        return MLx

    def apply_lumped_mass_matrix_rowsum(me, x):
        if isinstance(x, np.ndarray):
            MLx = me.mass_lumps_rowsum_numpy * x
        else:
            MLx = me.mass_lumps_rowsum_petsc * x
        return MLx

    def solve_lumped_mass_matrix_diagonal(me, x):
        if isinstance(x, np.ndarray):
            iMLx = x / me.mass_lumps_diagonal_numpy
        else:
            iMLx = me.numpy2petsc(me.petsc2numpy(x) / me.mass_lumps_diagonal_numpy)
        return iMLx

    def solve_lumped_mass_matrix_rowsum(me, x):
        if isinstance(x, np.ndarray):
            iMLx = x / me.mass_lumps_rowsum_numpy
        else:
            iMLx = me.numpy2petsc(me.petsc2numpy(x) / me.mass_lumps_rowsum_numpy)
        return iMLx

    def solve_mass_matrix_PCG(me, y, **kwargs):
        if isinstance(y, np.ndarray):
            y_numpy = y
        else:
            y_numpy = me.petsc2numpy(y)

        x_numpy, info = spla.cg(me.apply_mass_matrix_linop, y_numpy,
                                M=me.solve_lumped_mass_matrix_diagonal_linop, **kwargs)

        if isinstance(y, np.ndarray):
            x = x_numpy
        else:
            x = me.numpy2petsc(x_numpy)

        return x

    def get_solve_mass_matrix_PCG_linop(me, **kwargs):
        f = lambda y: me.solve_mass_matrix_PCG(y, **kwargs)
        return spla.LinearOperator((me.dim(), me.dim()), matvec=f, rmatvec=f)




# mesh = dl.UnitSquareMesh(10,11)
# Vh = FunctionSpaceEnhanced(mesh, 'CG', 2)
#
# z = np.random.randn(Vh.dim())
# y1 = Vh.apply_mass_matrix(z)
# y2 = Vh.mass_matrix_scipy * z
#
# z_petsc = Vh.get_vector()
# z_petsc[:] = z
#
# y3_petsc = Vh.apply_mass_matrix(z_petsc)
#
# np.linalg.norm(y3_petsc[:] - y2)
#
# LU_solve_err = np.linalg.norm(Vh.apply_mass_matrix(Vh.solve_mass_matrix_LU(z)) - z) / np.linalg.norm(z)
# print('LU_solve_err=', LU_solve_err)
#
# PCG_solve_err = np.linalg.norm(Vh.apply_mass_matrix(Vh.solve_mass_matrix_PCG(z,tol=1e-15)) - z) / np.linalg.norm(z)
# print('PCG_solve_err=', PCG_solve_err)