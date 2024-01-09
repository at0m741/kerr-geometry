import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.interpolate import interp1d
from astropy import constants as const
from astropy import units as u

def schwarzschild_radius(radius, mass):
    G = 6.67430e-11
    c = 299792458
    r = radius
    return 2 * G * mass / (c**2)

def schwarzschild_metric(radius, dt, dr, dtheta, dphi, n_iterations):
    mass = 1.989e65
    G = 6.67430e-11
    c = 299792458
    theta = np.pi / 2
    rs = schwarzschild_radius(radius, mass)

    x = np.sqrt(radius ** 2 + 1) * np.sin(np.pi / 2) * np.sin(np.pi * 2)
    y = np.sqrt(radius ** 2 + 1) * np.sin(np.pi / 2) * np.sin(np.pi * 2)
    x1 = np.linspace(-radius, radius, 2000)
    y1 = np.linspace(-radius, radius, 2000)
    X, Y = np.meshgrid(x1, y1)

    metric = np.zeros_like(X)
    for _ in tqdm(range(n_iterations)):
        dt += 0.1
        dr += 0.1
        dtheta += 0.1
        dphi += 0.1
        metric += (1 - rs / np.sqrt(X**2 + Y**2)) * c**2 * dt**2 - (1 - rs / np.sqrt(X**2 + Y**2))**(-1) * dr**2 - (np.sqrt(X**2 + Y**2))**2 * (np.sin(np.pi * np.sqrt(X**2 + Y**2) / 2))**2 * dtheta**2 - (np.sqrt(X**2 + Y**2))**2 * (np.sin(np.pi * np.sqrt(X**2 + Y**2) / 2))**2 * np.sin(theta)**2 * dphi**2
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.text2D(0.0, 0.95, "r = {}".format(radius), transform=ax.transAxes)
    ax.set_title("Schwarzschild metric")
    ax.plot_surface(X, Y, metric, cmap='viridis')
    plt.style.use("dark_background")
    return metric

def kerr_metric(radius, mass, dt, dr, dtheta, dphi, n_iterations):
      global G, rH, ax, x1, y1, inner, outer, inner_ergo, outer_ergo
      x = np.sqrt(radius ** 2 + 1) * np.sin(np.pi / 2) * np.sin(np.pi * 2)
      y = np.sqrt(radius ** 2 + 1) * np.sin(np.pi / 2) * np.sin(np.pi * 2)
      x1 = np.linspace(-x, x, 100)
      y1 = np.linspace(-y, y, 100)
      X, Y = np.meshgrid(x1, y1)
      rs = schwarzschild_radius(radius, mass)
      radius = np.sqrt(X**2 + Y**2)
      a = 1
      G = 6.67430e-11
      c = 299792458
      theta = np.pi / 2
      sigma = radius ** 2 + a ** 2 * np.cos(theta) ** 2
      delta = rs * radius + a ** 2
      metric = np.zeros_like(X)
      rH = rs + np.sqrt(rs ** 2 - 4* a ** 2) / 2
      print("rH: {}".format(rH))
      for _ in tqdm(range(n_iterations)):
          dphi += 0.1
          dt += 0.1
          dtheta += 0.1
          dr += 0.1
          X, Y = np.meshgrid(x, y)
          X += 0.1
          Y += 0.1
          metric += (1 - rs / sigma) * c**2 * dt**2 - ((2 *rs * a * radius * (np.sin(theta)) ** 2) / sigma) * c * dt * dphi - (sigma / delta) * dr**2 + dtheta ** 2 + (radius ** 2 + a **2 + (rs * a ** 2 * radius * (np.sin(theta)) ** 2) / sigma) * (np.sin(theta)) ** 2 * dphi ** 2



      return metric


def christoffel_kerr(radius, mass):
     
     global G, c, christoffel_symbols, sigma, gamma_phirphi, gamma_trt, gamma_phrt, gamma_tthetat, gamma_phittheta, gamma_trphi, gamma_tthetaphi, gamma_phithetaphi, gamma_rtt, gamma_rtphi, gamma_thetatt, gamma_thetatphi, gamma_rrr, gamma_rthetatheta, gamma_thetarthta, gamma_rrtheta, gamma_thetarr, gamma_rphiphi, gamma_thetaphiphi
     G = 6.67430e-11
     c = 299792458
     theta = np.pi / 2
     
     rs = schwarzschild_radius(radius, mass)
     a = 1
     delta = rs * radius + a ** 2
     sigma = radius ** 2 + a ** 2 * np.cos(theta) ** 2
     gamma_trt = (mass * (radius ** 2 + a ** 2)) / delta * radius ** 2
     gamma_phrt = (mass * a) / delta * radius ** 2
     gamma_tthetat = - 2 * mass * a ** 2 * np.cos(theta) / radius ** 3
     gamma_phittheta = - (2 * mass * a * np.cos(theta)) / radius ** 3
     gamma_trphi = - (mass * a * (3 * radius ** 3 + a ** 2 )) / delta * radius ** 2
     gamma_phirphi = (radius * (radius - 2 * mass * radius) - mass * a ** 2) / delta * radius ** 2
     gamma_tthetaphi = (2 * mass * a ** 3 * np.cos(theta)) / radius ** 3
     gamma_phithetaphi = (1 + (2 * mass * a ** 2 ) / radius ** 3) * np.cos(theta)
     gamma_rtt = (mass * delta) / radius ** 4
     gamma_rtphi = - (mass * a * delta) / radius ** 4
     gamma_thetatt = - (2 * mass * a ** 2 * np.cos(theta)) / radius ** 5
     gamma_thetatphi = - (2 * mass * a * (radius ** 2 + a ** 2) * np.cos(theta)) / radius ** 5
     gamma_rrr = (1 / radius) + (mass - radius / delta)
     gamma_rthetatheta = - (delta / radius)
     gamma_thetarthta = 1 / radius
     gamma_rrtheta = - ((a **2 * np.cos(theta)) / radius ** 2)
     gamma_thetarr = (a ** 2 * np.cos(theta)) / radius ** 2 * delta
     gamma_rphiphi = (delta * (mass * a ** 2 - radius ** 3)) / radius ** 4
     gamma_thetaphiphi = - (np.cos(theta) * (delta * radius ** 4 + 2 * mass * radius * (radius ** 2 + a ** 2) ** 2)) / radius ** 6

     return gamma_phirphi, gamma_trt, gamma_phrt, gamma_tthetat, gamma_phittheta, gamma_trphi, gamma_tthetaphi, gamma_phithetaphi, gamma_rtt, gamma_rtphi, gamma_thetatt, gamma_thetatphi, gamma_rrr, gamma_rthetatheta, gamma_thetarthta, gamma_rrtheta, gamma_thetarr, gamma_rphiphi, gamma_thetaphiphi

def metric_tensor():
    global g, theta, rs
    rs = schwarzschild_radius(radius, mass)
    theta = np.pi / 2
    g = np.zeros((4, 4))
    g[0][0] = -(1 - rs / radius)
    g[0][1] = 0
    g[0][2] = 0
    g[0][3] = 0
    g[1][0] = 0
    g[1][1] = (1 - rs / radius)
    g[1][2] = 0
    g[1][3] = 0
    g[2][0] = 0
    g[2][1] = 0
    g[2][2] = 0
    g[2][3] = radius ** 2
    g[3][0] = 0
    g[3][1] = 0
    g[3][2] = 0
    g[3][3] = radius ** 2 * np.sin(theta) ** 2
    return g

def compute_riemann_tensor():
    global christoffel_symbols, riemann_tensor, gadrient
    riemann_tensor = np.zeros((4, 4, 4, 4))
    G = 6.67430e-11
    c = 299792458
    theta = np.pi / 2
    r = radius
    M = mass
    christoffel_symbols = np.zeros((4, 4, 4, 4))
    christoffel_symbols[0][1][0][1] = (2 * G * M) / (r**2 * (-2 * G * M + c**2 * r))
    christoffel_symbols[0][1][1][0] = -((2 * G * M) / (r**2 * (-2 * G * M + c**2 * r)))
    christoffel_symbols[0][2][0][3] = -(G * M) / (c**2 * r)
    christoffel_symbols[0][2][2][0] = (G * M) / (c**2 * r)
    christoffel_symbols[0][3][0][0] = - (G * M *np.sin(theta)**2) / (c**2 * r)
    christoffel_symbols[0][3][3][0] = (G * M * np.sin(theta)**2) / (c**2 * r)
    christoffel_symbols[1][0][0][1] = (2 * G * M * (-2 * G * M + c**2 * r)) / (c**4 * r**4)
    christoffel_symbols[1][0][1][0] = -((2 * G * M * (-2 * G * M + c**2 * r)) / (c**4 * r**4))
    christoffel_symbols[1][2][1][3] = -((G * M) / (c**2 * r))
    christoffel_symbols[1][2][2][1] = (G * M) / (c**2 * r)
    christoffel_symbols[1][3][1][3] = -(G * M * np.sin(theta)**2) / (c**2 * r)
    christoffel_symbols[1][3][3][1] = (G * M * np.sin(theta)**2) / (c**2 * r)
    christoffel_symbols[2][0][0][2] = (G * M * (2 * G * M + c**2 * r)) / (c**4 * r**4)
    christoffel_symbols[2][0][2][0] = -((G * M * (2 * G * M + c**2 * r)) / (c**4 * r**4))
    christoffel_symbols[2][1][1][2] = (G * M) / (r**2 * (-2 * G * M + c**2 * r))
    christoffel_symbols[2][1][2][1] = (G * M ) / (r**2 * (-2 * G * M + c**2 * r))
    christoffel_symbols[2][3][2][3] = (2 * G * M * np.sin(theta)**2) / (c**2 * r)
    christoffel_symbols[2][3][3][2] = -(2 * G * M * np.sin(theta)**2) / (c**2 * r)
    christoffel_symbols[3][0][0][3] = (G * M * (2 * G * M - c**2 * r)) / (c**4 * r**4)
    christoffel_symbols[3][0][3][0] = (G * M * (-2 * G * M - c**2 * r)) / (c**4 * r**4)
    christoffel_symbols[3][1][1][3] = (G * M) / (r**2 * (-2 * G * M + c**2 * r))
    christoffel_symbols[3][1][3][2] = (G * M) / (r**2 * (2 * G * M + c**2 * r))
    christoffel_symbols[3][2][2][3] = -((2 * G * M) / (c**2 * r))
    christoffel_symbols[3][2][3][2] = (2 * G * M) / (c**2 * r)
    for rho in range(4):
        for sigma in range(4):
            for mu in range(4):
                for nu in range(4):
                    term1 = np.sum(np.gradient(christoffel_symbols[rho][nu][sigma], axis=0))
                    term2 = np.sum(np.gradient(christoffel_symbols[rho][mu][sigma], axis=0))
                    term3 = np.sum([np.gradient(christoffel_symbols[rho][lambda_][mu] * christoffel_symbols[lambda_][nu][sigma], axis=0) for lambda_ in range(4)])
                    term4 = np.sum([np.gradient(christoffel_symbols[rho][lambda_][nu] * christoffel_symbols[lambda_][mu][sigma], axis=0) for lambda_ in range(4)])
                    riemann_tensor[rho][sigma][mu][nu] = term1 - term2 + term3 - term4

    ricci_tensor = np.zeros((4, 4))
    for mu in range(4):
         for nu in range(4):
              ricci_tensor[mu][nu] = np.sum([riemann_tensor[rho][mu][rho][nu] for rho in range(4)])              

    print("Riemann tensor:\n {}\n".format(riemann_tensor))
    print("Ricci tensor:\n {}\n".format(ricci_tensor))
    ricci_scalar = np.sum([ricci_tensor[mu][nu] for mu in range(4) for nu in range(4)])
    print("Ricci scalar: {}\n".format(ricci_scalar))
    scalar_curvature = np.sum([np.sum([riemann_tensor[rho][sigma][mu][nu] for rho in range(4)]) for sigma in range(4) for mu in range(4) for nu in range(4)])
    print("Scalar curvature: \n {}\n".format(scalar_curvature))
    ricci_scalar_curvature = np.trace(ricci_tensor)
    print("Ricci scalar curvature: \n {}\n".format(ricci_scalar_curvature))
    einstein_tensor = ricci_tensor - (1 / 2) * ricci_scalar * metric_tensor()
    print("Einstein tensor: \n {}\n".format(einstein_tensor))

    return christoffel_symbols


def compute_kerr_metric_and_derivs(r, theta, M, a):
    c = 399792458 

    rs = schwarzschild_radius(r, M)
    sigma = r**2 + a**2 * np.cos(theta)**2
    delta = r**2 - rs * r + a**2

    g_tt = (1 - rs / sigma) * c**2
    g_tphi = -((2 * rs * a * r * (np.sin(theta)) ** 2) / sigma) * c
    g_rr = (sigma / delta)
    g_thetatheta = sigma
    g_phiphi = (r ** 2 + a ** 2 + (rs * a ** 2 * r * (np.sin(theta)) ** 2) / sigma) * (np.sin(theta)) ** 2
    metric = np.array([
        [g_tt, 0, 0, g_tphi],
        [0, -g_rr, 0, 0],
        [0, 0, -g_thetatheta, 0],
        [g_tphi, 0, 0, -g_phiphi]
    ])
    metric_derivs = np.zeros((4, 4, 4))
    eps = 1e-8 
    for i in range(4):
        for j in range(4):
            for k in range(4):
                metric_plus_eps = np.copy(metric)
                metric_plus_eps[i, j] += eps
                metric_minus_eps = np.copy(metric)
                metric_minus_eps[i, j] -= eps
                metric_derivs[i][j][k] = (metric_plus_eps[j, k] - metric_minus_eps[j, k]) / (2 * eps)

    print("Kerr metric tensor:\n {}\n".format(metric))
    print("Kerr metric derivatives:\n {}\n".format(metric_derivs))
    return metric, metric_derivs


def compute_metric_element(metric, r, theta, M, a, k):
    rs = schwarzschild_radius(r, M)
    sigma = r**2 + a**2 * np.cos(theta)**2
    delta = r**2 - rs * r + a**2

    if k == 0:
        return (1 - rs / sigma) * 399792458**2
    elif k == 1:
        return -(sigma / delta)
    elif k == 2:
        return -sigma
    elif k == 3:
        return (r ** 2 + a **2 + (rs * a ** 2 * r * (np.sin(theta)) ** 2) / sigma) * (np.sin(theta)) ** 2
    else:
        return 0.0 



def compute_christoffels_sym(G, M, r, c, theta):
    global christoffel_symbols, fig, ax, scatter_particles, scatter_black_hole
    christoffel_symbols = np.zeros((4, 4, 4, 4))

    formulas = {
        (0, 1, 0, 1): lambda: (2 * G * M) / (r**2 * (-2 * G * M + c**2 * r)),
        (0, 1, 1, 0): lambda: -((2 * G * M) / (r**2 * (-2 * G * M + c**2 * r))),
        (0, 2, 0, 3): lambda: -(G * M) / (c**2 * r),
        (0, 2, 2, 0): lambda: (G * M) / (c**2 * r),
        (0, 3, 0, 0): lambda: - (G * M *np.sin(theta)**2) / (c**2 * r),
        (0, 3, 3, 0): lambda: (G * M * np.sin(theta)**2) / (c**2 * r),
        (1, 0, 0, 1): lambda: (2 * G * M * (-2 * G * M + c**2 * r)) / (c**4 * r**4),
        (1, 0, 1, 0): lambda: -((2 * G * M * (-2 * G * M + c**2 * r)) / (c**4 * r**4)),
        (1, 2, 1, 3): lambda: -((G * M) / (c**2 * r)),
        (1, 2, 2, 1): lambda: (G * M) / (c**2 * r),
        (1, 3, 1, 3): lambda: -(G * M * np.sin(theta)**2) / (c**2 * r),
        (1, 3, 3, 1): lambda: (G * M * np.sin(theta)**2) / (c**2 * r),
        (2, 0, 0, 2): lambda: (G * M * (2 * G * M + c**2 * r)) / (c**4 * r**4),
        (2, 0, 2, 0): lambda: -((G * M * (2 * G * M + c**2 * r)) / (c**4 * r**4)),
        (2, 1, 1, 2): lambda: (G * M) / (r**2 * (-2 * G * M + c**2 * r)),
        (2, 1, 2, 1): lambda: (G * M ) / (r**2 * (-2 * G * M + c**2 * r)),
        (2, 3, 2, 3): lambda: (2 * G * M * np.sin(theta)**2) / (c**2 * r),
        (2, 3, 3, 2): lambda: -(2 * G * M * np.sin(theta)**2) / (c**2 * r),
        (3, 0, 0, 3): lambda: (G * M * (2 * G * M - c**2 * r)) / (c**4 * r**4),
        (3, 0, 3, 0): lambda: (G * M * (-2 * G * M - c**2 * r)) / (c**4 * r**4),
        (3, 1, 1, 3): lambda: (G * M) / (r**2 * (-2 * G * M + c**2 * r)),
        (3, 1, 3, 2): lambda: (G * M) / (r**2 * (2 * G * M + c**2 * r)),
        (3, 2, 2, 3): lambda: -((2 * G * M) / (c**2 * r)),
        (3, 2, 3, 2): lambda: (2 * G * M) / (c**2 * r)
    }

    for indices, formula in formulas.items():
        christoffel_symbols[indices] = formula()
    print("Christoffel symbols: \n {}\n".format(christoffel_symbols))

    return christoffel_symbols

def geodesic_equation(y, G, M, r, c, theta):
    global christoffel_symbols
    christoffel_symbols = compute_christoffels_sym(G, M, r, c, theta)
    dydt = np.zeros_like(y)

    return dydt


mass = 1.989e30 * 1.5e3
radius = 100
G = 6.67430e-11
c = 299792458
a = 1
theta = np.pi / 2
print("Schwarzschild radius: {}".format(radius))
dt, dr, dtheta, dphi = 0.1, 0.1, 0.1, 0.1
dt_inc, dr_inc, dtheta_inc, dphi_inc = 0.01, 0.01, 0.01, 0.01
n_iterations = 100
rs = schwarzschild_radius(radius, mass)

outer = rs + np.sqrt(rs ** 2 - a ** 2)
inner = rs - np.sqrt(rs ** 2 - a ** 2)
outer_ergo = rs + np.sqrt(rs ** 2 - a ** 2 * np.cos(theta) ** 2)
inner_ergo = rs - np.sqrt(rs ** 2 - a ** 2 * np.cos(theta) ** 2)
print("outer: {}".format(outer))
print("inner: {}".format(inner))
print("outer ergo: {}".format(outer_ergo))
print("inner ergo: {}".format(inner_ergo))


x = np.linspace(-radius, radius, 100)
y = np.linspace(-radius, radius, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2)) + radius ** 2 * np.cos(np.pi / 2)
sym = compute_christoffels_sym(G, mass, radius, c, np.pi / 2)
metric = kerr_metric(radius, mass, dt, dr, dtheta, dphi, n_iterations)
#metric2 = schwarzschild_metric(radius, dt, dr, dtheta, dphi, n_iterations)
symbols = christoffel_kerr(radius, mass)
compute_riemann_tensor()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

rs = schwarzschild_radius(radius, mass)
X, Y = np.meshgrid(x, y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-radius, radius)
ax.set_ylim(-radius, radius)
plt.title("Kerr spacetime", )
ax.plot_surface(X, Y, metric, cmap='hot', rstride=1, cstride=1)
ax.plot([0, radius], [0, 0], [0, 0], label='R', color='red', linewidth=4)
ax.set_xlim(-radius, radius)
ax.set_ylim(-radius, radius)
ax.text2D(0.0, 0.95, "r = {}".format(radius), transform=ax.transAxes)
radius_values = [radius for _ in range(n_iterations)]
christoffel_kerr(radius, mass)
ax.legend()

plt.show()

