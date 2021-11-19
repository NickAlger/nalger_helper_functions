import numpy as np
import matplotlib.pyplot as plt

# x0 = 0.0
# x0 = 0.25
# x0 = 1.0
# x0 = 2.0
x0 = -0.25
# x0 = -1.0

func = lambda x: np.exp(-0.5 * np.power(x - x0, 2)) * (x<=0)
# func = lambda x: np.exp(-0.5 * np.power(x - x0, 2)) * (x<=0) * (-2.5 <= x) # what happens if we truncate small values?
# func = lambda x: 0.5 * np.power(x - x0, 2) * (x<=0) # purely quadratic function
# func = lambda x: np.exp(-1. / (1. - np.power(x - x0,2))) * (np.abs(x-x0) < 1.) # compactly supported bump function

xx = np.linspace(-3, 0, 301)
ff = func(xx)

plt.figure()
plt.plot(xx, ff)

all_orders = [0,1,2,3]
for order in all_orders:
    kk = np.arange(order+1)
    # jj = np.arange(order+1) + 1 # Standard nodes, e.g., in textbook proof
    # jj = np.linspace(0., 1., order+1)
    jj = np.linspace(0.1, 1., order + 1)
    # jj = 1. + 0.5*(jj-1.) # squash/stretch nodes
    # jj = 1. / jj # invert nodes
    M = np.power(-jj[None,:], kk[:,None]) # Vandermonde matrix
    mu = np.linalg.solve(M, np.ones(order+1))

    yy = np.linspace(0, 3, 301)
    gg = np.zeros(len(yy))
    for i in range(len(yy)):
        y = yy[i]
        gg[i] = np.sum(func(-jj * y) * mu)

    plt.plot(yy,gg)

plt.legend(['f'] + ['order=' + str(x) for x in all_orders])