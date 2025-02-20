import numpy as np
import matplotlib.pyplot as plt

# Define spatial and time grid
Nx, Ny = 150, 150
Nt = 1000
L = 15.0
dx, dy = L / Nx, L / Ny
dt = 0.01

# Fluxon interaction parameters
m = 1.0
g = 1.0
attractive_potential = -0.5

# Initialize fluxonic atomic state
x = np.linspace(-L/2, L/2, Nx)
y = np.linspace(-L/2, L/2, Ny)
X, Y = np.meshgrid(x, y)
phi = np.exp(-(X**2 + Y**2)) * np.cos(4 * np.sqrt(X**2 + Y**2))  # Orbital-like initial condition
phi_old = np.copy(phi)
phi_new = np.zeros_like(phi)

# Simulation loop for atomic formation
for n in range(Nt):
    d2phi_dx2 = (np.roll(phi, -1, axis=0) - 2 * phi + np.roll(phi, 1, axis=0)) / dx**2
    d2phi_dy2 = (np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1)) / dy**2
    interaction_term = attractive_potential * phi
    phi_new = 2 * phi - phi_old + dt**2 * (d2phi_dx2 + d2phi_dy2 - m**2 * phi - g * phi**3 + interaction_term)
    phi_old = np.copy(phi)
    phi = np.copy(phi_new)

# Visualization
plt.figure(figsize=(10, 6))
plt.imshow(phi, cmap="inferno", extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(label="Fluxon Field Intensity")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("2D Fluxonic Atomic Structure (m=1, g=1)")
plt.savefig("matter_2d_plot.png")
plt.close()

print("Simulation done! Check matter_2d_plot.png")