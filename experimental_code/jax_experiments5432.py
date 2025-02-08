import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

A = np.random.randn(5,5)
A_jax = jnp.array(A)

x = np.random.randn(5,2)

def f(M, z):
    return 0.5 * z.T @ M @ z

# @jax.jit
def f_jax(M, z):
    return 0.5 * z.T @ M @ z

g_func_01 = jax.grad(f_jax, argnums=[0,1])

p = np.random.randn(5)
dA, dp = g_func_01(A,p)

jax.jvp(g_func_01, (A, p), (dA, dp))

J_func = jax.jacobian(g_func_01,argnums=[0,1])

((JAA, JAp), (JpA, Jpp)) = J_func(A, p)

@jax.jit
def g(x):
    s = 0
    for i in range(10):
        s += x[i]
    return x

@jax.jit # error
def add_one(M):
    M[0,0] += 1
    return M

M = np.zeros((2,2))
M2 = add_one(M)
print(M)
print(M2)

