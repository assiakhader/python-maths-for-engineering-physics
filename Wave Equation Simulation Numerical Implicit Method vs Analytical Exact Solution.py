import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
L = 1.0         # Length of the domain
T = 2.0         # Total time
h = 0.1         # Spatial step size
k = 0.1         # Time step size
c = 1.0         # Wave speed
lambda_ = (k**2) / (h**2)

Nx = int(L / h) + 1  # Number of spatial points
Nt = int(T / k)      # Number of time steps

x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Initial conditions
f = x * (1 - x)
u_initial = f
u_previous = np.zeros(Nx)  # u(x, -k) is assumed to be zero

# Numerical solution setup
u = np.zeros((Nt, Nx))
u[0, :] = u_initial  # Initial condition at t=0
u[1, :] = u_initial  # First time step (ut(x, 0) = 0)

# Time-stepping loop
for n in range(1, Nt-1):
    for i in range(1, Nx-1):
        u[n+1, i] = 2 * (1 - lambda_) * u[n, i] + lambda_ * (u[n, i+1] + u[n, i-1]) - u[n-1, i]

# Exact solution using d'Alembert's formula
def compute_bn(n):
    """Compute the Fourier sine coefficient b_n for f(x) = x(1-x)."""
    return (4 / (n**3 * np.pi**3)) * (1 - (-1)**n)

def Sf_s(x, terms=10):
    """Compute the Fourier sine series."""
    result = np.zeros_like(x)
    for n in range(1, terms + 1):
        result += compute_bn(n) * np.sin(n * np.pi * x)
    return result

u_exact = np.zeros((Nt, Nx))
for n in range(Nt):
    u_exact[n, :] = 0.5 * (Sf_s(x + t[n]) + Sf_s(x - t[n]))

# Plot the difference between numerical and exact solutions
X, T = np.meshgrid(x, t)  # Create grid for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, np.abs(u - u_exact), cmap='plasma')
ax.set_title('Difference: Numerical vs Exact Solution')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('|Numerical - Exact|')
plt.show()
