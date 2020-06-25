import numpy as np
import fenics
import matplotlib.pyplot as plt

n=40
num_timesteps = 3
dt = 1e-2

mesh = fenics.UnitSquareMesh(40,40)
V = fenics.FunctionSpace(mesh, 'CG', 1)

u = fenics.TrialFunction(V)
v = fenics.TestFunction(V)

mass_form = u * v * fenics.dx
stiffness_form = fenics.inner(fenics.grad(u), fenics.grad(v)) * fenics.dx

M = fenics.assemble(mass_form)
Z = fenics.assemble(mass_form + dt * stiffness_form)

Z_solver = fenics.LUSolver(Z)

def smooth_function(f):
    for k in range(num_timesteps):
        Z_solver.solve(f.vector(), M * f.vector())


w = fenics.Function(V)
w.vector()[:] = np.random.randn(V.dim())

plt.figure()
fenics.plot(w)
plt.title('unsmoothed')

smooth_function(w)

plt.figure()
fenics.plot(w)
plt.title('smoothed')
