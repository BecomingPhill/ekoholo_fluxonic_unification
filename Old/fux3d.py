import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Grid setup
Nx, Ny, Nz = 50, 50, 50
L, dt, Nt = 10.0, 0.01, 700
dx = L / Nx
x, y, z = [np.linspace(-L/2, L/2, N) for N in [Nx, Ny, Nz]]
X, Y, Z = np.meshgrid(x, y, z)

# Parameters
m, g = 1.0, 1.0
potentials = {'wave': -0.8, 'collapse': -1.5, 'shielding': -2.0, 'atomic': -0.5}

# Initial conditions
inits = {
    'wave': np.exp(-((X-5)**2 + Y**2 + Z**2)) * np.sin(6 * np.sqrt((X-5)**2 + Y**2 + Z**2)),
    'collapse': np.exp(-(X**2 + Y**2 + Z**2)),
    'shielding': np.exp(-((X-5)**2 + Y**2 + Z**2)) * np.sin(6 * np.sqrt((X-5)**2 + Y**2 + Z**2)),
    'atomic': np.exp(-(X**2 + Y**2 + Z**2)) * np.cos(4 * np.sqrt(X**2 + Y**2 + Z**2))
}

# Run simulation for each case
for case in ['wave', 'collapse', 'shielding', 'atomic']:
    phi = np.copy(inits[case])
    phi_old, phi_new = np.copy(phi), np.zeros_like(phi)
    
    for n in range(Nt):
        d2phi_dx2 = (np.roll(phi, -1, 0) - 2 * phi + np.roll(phi, 1, 0)) / dx**2
        d2phi_dy2 = (np.roll(phi, -1, 1) - 2 * phi + np.roll(phi, 1, 1)) / dx**2
        d2phi_dz2 = (np.roll(phi, -1, 2) - 2 * phi + np.roll(phi, 1, 2)) / dx**2
        potential_term = potentials[case] * phi
        if case == 'shielding':  # Add barrier
            potential_term += np.where(np.abs(X) < 1, -2.0, 0) * phi
        phi_new = 2 * phi - phi_old + dt**2 * (d2phi_dx2 + d2phi_dy2 + d2phi_dz2 - m**2 * phi - g * phi**3 + potential_term)
        phi_old, phi = np.copy(phi), np.copy(phi_new)
    
    # Save plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=phi.flatten(), cmap='inferno', s=1)
    ax.set_title(f"3D Fluxonic {case.capitalize()}")
    plt.savefig(f"{case}_plot.png")
    plt.close()

print("Simulations done! Check the PNG files.")