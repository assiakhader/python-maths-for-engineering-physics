import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Set parameters
L = 1.0         # Length of the rod
T = 2.0         # Total time
h = 0.1         # Spatial step size
k = 0.1         # Time step size
alpha = 0.25    # Diffusion coefficient

Nx = int(L / h) + 1  # Number of spatial points
Nt = int(T / k)      # Number of time steps

x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Step 2: Initial condition
u_initial = np.zeros(Nx)
for i in range(Nx):
    if x[i] <= 0.5:
        u_initial[i] = 20 * x[i]
    else:
        u_initial[i] = 20 * (1 - x[i])

# Step 3: Crank-Nicholson method
sigma = alpha * k / h**2

# Create the coefficient matrix
A = np.zeros((Nx-2, Nx-2))
for i in range(Nx-2):
    A[i, i] = 1 + sigma
    if i > 0:
        A[i, i-1] = -sigma / 2
    if i < Nx-3:
        A[i, i+1] = -sigma / 2

B = np.zeros((Nx-2, Nx-2))
for i in range(Nx-2):
    B[i, i] = 1 - sigma
    if i > 0:
        B[i, i-1] = sigma / 2
    if i < Nx-3:
        B[i, i+1] = sigma / 2

# Time evolution
u = np.zeros((Nt, Nx))
u[0, :] = u_initial

for n in range(0, Nt-1):
    b = B @ u[n, 1:-1]
    u[n+1, 1:-1] = np.linalg.solve(A, b)

# Step 4: Fourier series solution
def fourier_series(x, t, terms=10):
    result = np.zeros_like(x)
    for n in range(1, terms + 1):
        if n % 2 == 1:  # Only odd terms contribute
            lambda_n = (n * np.pi / L)**2
            bn = (80 / (n**3 * np.pi**3)) * (1 - (-1)**n)
            result += bn * np.sin(n * np.pi * x / L) * np.exp(-alpha * lambda_n * t)
    return result

u_fourier = np.zeros((Nt, Nx))
for n in range(Nt):
    u_fourier[n, :] = fourier_series(x, t[n])

# Step 5: Plot results
# Numerical solution
X, T = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u, cmap='viridis')
ax.set_title('Numerical Solution')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x, t)')
plt.show()

# Difference between numerical and Fourier series solution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, np.abs(u - u_fourier), cmap='plasma')
ax.set_title('Difference: Numerical vs Fourier')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('|Numerical - Fourier|')
plt.show()
# Compute the Fourier series solution
def fourier_series(x, t, terms=10):
    result = np.zeros_like(x)
    for n in range(1, terms + 1):
        if n % 2 == 1:  # Only odd terms contribute
            lambda_n = (n * np.pi)**2
            bn = (80 / (n**3 * np.pi**3)) * (1 - (-1)**n)
            result += bn * np.sin(n * np.pi * x) * np.exp(-0.25 * lambda_n * t)
    return result

u_fourier = np.zeros((Nt, Nx))
for n in range(Nt):
    u_fourier[n, :] = fourier_series(x, t[n])

# Plot the difference |numerical - Fourier series|
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, np.abs(u - u_fourier), cmap='plasma')
ax.set_title('Difference: Numerical vs Fourier Series Solution')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('|Numerical - Fourier|')
plt.show()
