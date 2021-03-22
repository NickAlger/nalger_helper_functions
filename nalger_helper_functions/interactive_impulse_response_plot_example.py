import dolfin as dl
from nalger_helper_functions import interactive_impulse_response_plot


n = 40
mesh = dl.UnitSquareMesh(n,n)
V = dl.FunctionSpace(mesh, 'CG', 1)
u_trial = dl.TrialFunction(V)
v_test = dl.TestFunction(V)
a = dl.inner(dl.grad(u_trial), dl.grad(v_test)) * dl.dx + u_trial * v_test * dl.dx
A_fenics = dl.assemble(a)
x_fenics = dl.Function(V)

solve_A_fenics = dl.LUSolver(A_fenics)
def apply_invA(b_petsc):
    solve_A_fenics.solve(x_fenics.vector(), b_petsc)
    return x_fenics.vector()

interactive_impulse_response_plot(apply_invA, V)
