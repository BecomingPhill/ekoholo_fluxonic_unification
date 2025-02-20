import numpy as np
import matplotlib.pyplot as plt

# Define spatial and time grid
Nx, Ny = 150, 150
Nt = 1500
L = 15.0
dx, dy = L / Nx, L / Ny
dt = 0.01

# Fluxon interaction parameters
m = 1.0
g = 1.0
gravitational_potential = -1.0
rotation_potential = -0.8

# Initialize fluxonic gravitational core
x = np.linspace(-L/2, L/2, Nx)
y = np.linspace(-L/2, L/2, Ny)
X, Y = np.meshgrid(x, y)
phi = np.exp(-(X**2 + Y**2)) * np.sin(4 * np.arctan2(Y, X))  # Rotating initial condition
phi_old = np.copy(phi)
phi_new = np.zeros_like(phi)

# Simulation loop
for n in range(Nt):
    d2phi_dx2 = (np.roll(phi, -1, axis=0) - 2 * phi + np.roll(phi, 1, axis=0)) / dx**2
    d2phi_dy2 = (np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1)) / dy**2
    gravity_term = gravitational_potential * np.sqrt(X**2 + Y**2) * phi
    rotation_term = rotation_potential * (X * np.roll(phi, -1, axis=1) - Y * np.roll(phi, -1, axis=0))  # Completed rotation
    phi_new = 2 * phi - phi_old + dt**2 * (d2phi_dx2 + d2phi_dy2 - m**2 * phi - g * phi**3 + gravity_term + rotation_term)
    phi_old = np.copy(phi)
    phi = np.copy(phi_new)

# Visualization
plt.figure(figsize=(10, 6))
plt.imshow(phi, cmap="inferno", extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(label="Fluxon Field Intensity")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("2D Fluxonic Gravitational Waves (m=1, g=1)")
plt.savefig("gravity_2d_plot.png")
plt.close()

print("Simulation done! Check gravity_2d_plot.png")