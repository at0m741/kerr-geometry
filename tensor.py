import sympy as sp
import numpy as np
from math import pi as M_PI
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Tensor Calculus')
    parser.add_argument('--M', type=float, default=1.989e30)
    parser.add_argument('--G', type=float, default=6.674e-11)
    parser.add_argument('--c', type=float, default=299792458)
    parser.add_argument('--theta', type=float, default=0)
    parser.add_argument('--r', type=float, default=1.5e11)
    parser.add_argument('--a', type=float, default=1)
    parser.add_argument('--t', type=float, default=0)
    args = parser.parse_args()
    return args


def aff(M):
    le = [([len(str(i)) for i in k]) for k in M]
    l = [max([le[k][i] for k in range(len(M))]) for i in range(len(M[0]))]
    for k in range(len(M)):
        print('[', end="")
        for j in range(len(M[k])):
            print(M[k][j], end=" " * (1 + l[j] - len(str(M[k][j]))))
        print(']')


def schwarzschild_metric(M, G, c, theta, r):
    Rs = 2 * G * M / (c ** 2)

    g = sp.Matrix([[1 - Rs / r, 0, 0, 0],
                   [0, -1 / (1 - Rs / r), 0, 0],
                   [0, 0, -r ** 2, 0],
                   [0, 0, 0, -r ** 2 * sp.sin(theta) ** 2]])
    return g


def kerr_metric(M, G, c, theta, a, r):
    Rs = 2 * G * M / (c ** 2)
    a = 1
    Delta = r ** 2 - Rs * r + a ** 2
    rho = sp.sqrt(r ** 2 + a ** 2 * sp.cos(theta) ** 2)
    g = [[1 - Rs * rho ** 2 / Delta, 0, 0, -Rs * a * rho ** 2 * sp.sin(theta) ** 2 / Delta],
         [0, -Delta / Delta, 0, 0],
         [0, 0, -rho ** 2, 0],
         [-Rs * a * rho ** 2 * sp.sin(theta) ** 2 / Delta, 0, 0,
          -rho ** 2 * sp.sin(theta) ** 2 * (r ** 2 + a ** 2 + Rs * a ** 2 * rho ** 2 * sp.sin(theta) ** 2 / Delta)]]
    return g


r = sp.Symbol('r')
theta = sp.Symbol('theta')
phi = sp.Symbol('phi')
t = sp.Symbol('t')
k = [t, r, theta, phi]
M = sp.Symbol('M')
G = sp.Symbol('G')
c = sp.Symbol('c')
Rs = sp.Symbol('Rs')
a = sp.Symbol('a')
rho = sp.Symbol('rho')
Delta = sp.Symbol('Delta')

g = [[(1 - (2 * G * M) / r), 0, 0, 0],
     [0, (1 / (1 - 2 * G * M / r)), 0, 0],
     [0, 0, r ** 2, 0],
     [0, 0, 0, r ** 2 * sp.sin(theta) ** 2]]

print("\u0332".join("Schwarzschild Metric: \n"))
aff(g)
print("\n")
g2 = sp.Matrix(([g[0][0], g[1][0], g[2][0], g[3][0]],
                [g[0][1], g[1][1], g[2][1], g[3][1]],
                [g[0][2], g[1][2], g[2][2], g[3][2]],
                [g[3][0], g[3][1], g[3][2], g[3][3]]))
g_inv = g2.inv()

print("\u0332".join("Metric Tensor: \n"))
for _ in g:
    print(_)
print("\n")


def aff_christoffel_numeric(Christoffel_numeric):
    for i, matrix in enumerate(Christoffel_numeric):
        print(f"Christoffel for {k[i]}:\n")
        formatted_matrix = np.array(matrix)
        np.set_printoptions(precision=3, suppress=True)
        print(formatted_matrix)
        print("\n")


def Christoffel():
    global Christo, Christo_t, Christo_r, Christo_o, Christo_p, Christoffel

    Christo = [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0], ]
    Christo_t = [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0], ]
    Christo_r = [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0], ]
    Christo_o = [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0], ]
    Christo_p = [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0], ]
    Christoffel = []

    print("\u0332".join("Christofell symbols: \n"))
    for m in range(4):
        for l in range(4):
            for i in range(4):
                for j in range(4):
                    Christo[i][j] = 1 / 2 * g_inv[l, m] * (
                            sp.diff(g[l][i], k[j]) + sp.diff(g[j][l], k[i]) - sp.diff(g[i][j], k[l]))
                    if Christo[i][j] != 0:
                        print("Γ^", i, j, "_", l, " = ", Christo[i][j])
                        if m == 0:
                            Christo_t[i][j] = Christo[i][j]
                        if m == 1:
                            Christo_r[i][j] = Christo[i][j]
                        if m == 2:
                            Christo_o[i][j] = Christo[i][j]
                        if m == 3:
                            Christo_p[i][j] = Christo[i][j]
    Cristoffel = [Christo_t, Christo_r, Christo_o, Christo_p]
    print("\n")
    for i in range(4):
        print("Christoffel for", k[i], ":")
        print("\n")
        aff(Cristoffel[i])
        print("\n")

    return Cristoffel


def christoffel_compute(r_value, theta_value, phi_value, t_value, M, G, c):
    global Christo, Christo_t, Christo_r, Christo_o, Christo_p, Christoffel

    r, theta, phi, t = sp.symbols('r theta phi t')
    x = [r, theta, phi, t]

    g = [[(1 - (2 * G * M) / r), 0, 0, 0],
         [0, (1 / (1 - 2 * G * M / r)), 0, 0],
         [0, 0, r ** 2, 0],
         [0, 0, 0, r ** 2 * sp.sin(theta) ** 2]]

    g2 = sp.Matrix(([g[0][0], g[1][0], g[2][0], g[3][0]],
                    [g[0][1], g[1][1], g[2][1], g[3][1]],
                    [g[0][2], g[1][2], g[2][2], g[3][2]],
                    [g[3][0], g[3][1], g[3][2], g[3][3]]))

    g_inv = g2.inv()
    Christo_t = [[0 for _ in range(4)] for _ in range(4)]
    Christo_r = [[0 for _ in range(4)] for _ in range(4)]
    Christo_o = [[0 for _ in range(4)] for _ in range(4)]
    Christo_p = [[0 for _ in range(4)] for _ in range(4)]
    print("\u0332".join("Christofell symbols: \n"))
    for l in range(4):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    Christo_symbol = 1 / 2 * g_inv[i, l] * (
                            sp.diff(g[l][j], x[k]) + sp.diff(g[l][k], x[j]) - sp.diff(g[j][k], x[l]))
                    Christo_symbol = Christo_symbol.subs({r: r_value, theta: theta_value, phi: phi_value, t: t_value})
                    Christo_symbol = Christo_symbol.evalf()
                    if Christo_symbol != 0:
                        print("Γ^", i, j, "_", l, " = ", Christo_symbol)
                        if l == 0:
                            Christo_t[i][j] = Christo_symbol
                        if l == 1:
                            Christo_r[i][j] = Christo_symbol
                        if l == 2:
                            Christo_o[i][j] = Christo_symbol
                        if l == 3:
                            Christo_p[i][j] = Christo_symbol

    Christoffel = [Christo_t, Christo_r, Christo_o, Christo_p]

    return Christoffel


def Riemann_tensor_compute(Christoffel):
    t, r, theta, phi = sp.symbols('t r theta phi')
    x = [t, r, theta, phi]
    R = [[[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    R[i][j][k][l] = sp.diff(Christoffel[i][j][l], x[k]) - sp.diff(Christoffel[i][j][k], x[l])
    print("\u0332".join("Riemann tensor: \n"))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    if R[i][j][k][l] != 0:
                        print("R^", i, j, "_", k, l, " = ", R[i][j][k][l])
    print("\n")
    print("\u0332".join("Riemann tensor matrix: \n"))
    print(R)
    return R


G = 6.674e-11
c = 299792458
M = 1.989e30
r = 1.5e11
theta = M_PI / 2
phi = M_PI * 2
t = 10

args = parse_args()
cr = Christoffel()
print("\n")
cr1 = christoffel_compute(r, theta, phi, t, M, G, c)
print("\n")
print("\u0332".join("Christofell symbols matrix: \n"))
print(cr1)

print("\n")

R = Riemann_tensor_compute(cr1)
