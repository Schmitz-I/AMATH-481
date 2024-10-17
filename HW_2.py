import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

tol = 10e-6
L = 4
xspan = np.arange(-L, L + 0.1, 0.1)
eps_start = 0.1

col = ['r', 'b', 'g', 'c', 'm', 'k'] # eigenfunc colors

eigenvalues = []
eigenfunctions = []

def RHS(y, x, En):
    return [y[1], (x**2 - En) * y[0]]

for modes in range(1, 6): # begin mode loop
    epsilon = eps_start # initial value of epsilon
    d_eps = 0.2 # default step size of epsilon

    for _ in range(1000): # begin convergence loop for epsilon
        Y0 = [1, np.sqrt(L**2 - epsilon)]
        y = odeint(RHS, Y0, xspan, args=(epsilon,))

        if abs(y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) < tol: # check for convergence
            eigenvalues.append(epsilon)
            break # get out of convergence loop

        if ((-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0])) > 0:
            epsilon += d_eps
        else:
            epsilon -= d_eps
            d_eps = d_eps / 2
    eps_start = epsilon + 0.1 # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], xspan) # calculate the normalization
    func = abs(y[:, 0]/ np.sqrt(norm))

    eigenfunctions.append(func)

    plt.plot(xspan, func, col[modes - 1])

A2 = eigenvalues
A1 = eigenfunctions
A1 = np.array(A1).T