import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants and Parameters
G = 6.67430e-11  # Gravitational constant
c = 299792458    # Speed of light
M = 1.5e30 * 1.5e3 # Mass of the black hole
a = 1  # Spin parameter
Rs = 2 * G * M / c**2

theta = np.linspace(0, np.pi / 2, 100)
phi = np.linspace(0, np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# Ergosphere equations
outer = Rs + np.sqrt(Rs ** 2 - a ** 2)
inner = Rs - np.sqrt(Rs ** 2 - a ** 2)
outer_ergo = Rs + np.sqrt(Rs**2 - a**2 * np.cos(theta)**2)
inner_ergo = Rs - np.sqrt(Rs**2 - a**2 * np.cos(theta)**2)
Ergosphere = np.sqrt(Rs**2 - a**2 * np.cos(theta)**2)

print("outer :\n", outer)
print("inner : \n", inner)
print("outer ergo \n", outer_ergo)
print("inner ergo \n", inner_ergo)
print(Rs)
print(a)


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z
ring_radius = np.sqrt(Rs**2 - a**2)
x_oe, y_oe, z_oe = spherical_to_cartesian(outer_ergo, theta, phi)
x_ie, y_ie, z_ie = spherical_to_cartesian(inner_ergo, theta, phi)
x_o, y_o, z_o = spherical_to_cartesian(outer, theta, phi)
x_i, y_i, z_i = spherical_to_cartesian(inner, theta, phi)
x_e, y_e, z_e = spherical_to_cartesian(Ergosphere, theta, phi)
phi_ring = np.linspace(0, 2*np.pi, 100)
x_ring = ring_radius * np.cos(phi_ring)
y_ring = ring_radius * np.sin(phi_ring)
z_ring = np.zeros_like(x_ring)

phi_equatorial = np.linspace(0, 2 * np.pi, 100)
outer_radius_equatorial = Rs + np.sqrt(Rs**2 - a**2)
inner_radius_equatorial = Rs - np.sqrt(Rs**2 - a**2)
outer_ergo_radius_equatorial = Rs + np.sqrt(Rs**2 - a**2 * np.cos(np.pi / 2)**2)
inner_ergo_radius_equatorial = Rs - np.sqrt(Rs**2 - a**2 * np.cos(np.pi / 2)**2)
ring_radius = np.sqrt(Rs**2 - a**2)

# Facteur d'asymétrie
asymmetry_factor = 0.9  # Modifier ce facteur pour changer l'ovale

# Coordonnées dans le plan équatorial
x_outer = outer_radius_equatorial * np.cos(phi_equatorial)
y_outer = outer_radius_equatorial * np.sin(phi_equatorial) * asymmetry_factor
x_inner = inner_radius_equatorial * np.cos(phi_equatorial)
y_inner = inner_radius_equatorial * np.sin(phi_equatorial) * asymmetry_factor
x_outer_ergo = outer_ergo_radius_equatorial * np.cos(phi_equatorial)
y_outer_ergo = outer_ergo_radius_equatorial * np.sin(phi_equatorial) * asymmetry_factor
x_inner_ergo = inner_ergo_radius_equatorial * np.cos(phi_equatorial)
y_inner_ergo = inner_ergo_radius_equatorial * np.sin(phi_equatorial) * asymmetry_factor

# Création du graphique de coupe horizontale
plt.figure(figsize=(8, 8))
plt.plot(x_outer, y_outer * 0.5, label='Outer Horizon', color='green')
plt.plot(x_inner, y_inner, label='Inner Horizon', color='black')
plt.plot(x_outer_ergo, y_outer_ergo, label='Outer Ergosphere', color='blue')
plt.plot(x_inner_ergo, y_inner_ergo * 0.5, label='Inner Ergosphere', color='red')
plt.plot(ring_radius * np.cos(phi_equatorial), ring_radius * np.sin(phi_equatorial) * asymmetry_factor, label='Ring Singularity', color='black', linestyle='--')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Horizontal Cut of the Kerr Black Hole (Oval Representation)')
plt.legend()
plt.axis('equal')
plt.show()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot full surfaces
ax.plot_surface(x_oe, y_oe, z_oe, color='blue', alpha=0.6, label='Outer Ergosphere')
ax.plot_surface(x_oe, y_oe, -z_oe, color='blue', alpha=0.6)
ax.plot_surface(x_ie, y_ie, z_ie, color='red', alpha=0.6, label='Inner Ergosphere')
ax.plot_surface(x_ie, y_ie, -z_ie, color='red', alpha=0.6)
ax.plot_surface(x_o, y_o, z_o, color='green', alpha=0.4, label='Outer Event Horizon')
ax.plot_surface(x_o, y_o, -z_o, color='green', alpha=0.4)
ax.plot_surface(x_i, y_i, z_i, color='black', alpha=0.8, label='Inner Event Horizon')
ax.plot_surface(x_i, y_i, -z_i, color='black', alpha=0.8)
ax.plot(x_ring, y_ring, z_ring, color='black', linewidth=2, label='Ring Singularity')
ax.plot_surface(x_e, y_e, z_e, color='orange', alpha=1, label='Ergosphere')
ax.plot_surface(x_e, y_e, -z_e, color='orange', alpha=1)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Full Visualization of Kerr Black Hole Ergospheres')

plt.show()