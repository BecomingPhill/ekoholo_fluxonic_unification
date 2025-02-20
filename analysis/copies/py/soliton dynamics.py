import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 20.0  # Spatial domain size
Nx = 200  # Grid points
dx = L / Nx
dt = 0.01  # Time step
Nt = 500  # Time iterations
m = 1.0  # Mass parameter
g = 1.0  # Nonlinearity coefficient

# Initialize fields
x = np.linspace(-L / 2, L / 2, Nx)
phi = np.tanh(x / np.sqrt(2))  # Initial soliton profile
phi_old = np.tanh((x - 0.3 * dt) / np.sqrt(2))  # Slightly shifted for dynamics
phi_new = np.zeros_like(phi)

# Time evolution
for n in range(Nt):
    d2phi_dx2 = (np.roll(phi, -1) - 2 * phi + np.roll(phi, 1)) / dx**2
    phi_new = 2 * phi - phi_old + dt**2 * (d2phi_dx2 - m**2 * phi - g * phi**3)
    phi_old = np.copy(phi)
    phi = np.copy(phi_new)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(x, phi, label="Final Soliton Profile")
plt.xlabel("x")
plt.ylabel("Amplitude")
plt.title("1D Soliton Evolution (m=1, g=1)")
plt.legend()
plt.savefig("soliton_plot.png")
plt.close()

print("Simulation done! Check soliton_plot.png")