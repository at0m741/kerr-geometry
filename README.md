# kerr-geometry
Still in developpement, but a cool Kerr Metric plotter with Riemann and Ricci Tensor solver (nedd to improve and get a real Partial differentiation solver)

# Plot

The plot only use the kerr metric atm. This metric decribe a rotating Black Hole that depend of cinetic moment :

```math
a = J/ m^2
```
this metric took this form :

```math
G_{\mu \nu}dx^\mu dx^\nu=-(1-{2GMr\over c^2p^2})c^2dt^2-{4GMac\sin^2\theta\over c^2p^2}cdtd\phi+{p^2\over\Delta}dr^2+p^2d\theta^2+(r^2+a^2+{2GMa^2r\sin^2\theta\over c^2p^2})\sin^2\theta d\phi^2
```

# about Tensor Solver

This part is ot complete for now and still in developpement. First I solve the Riemann tensor :

```math
  R{^\alpha_{\beta\mu\nu}} = \partial_\mu\Gamma{^\alpha_{\beta\nu}}-\partial_\nu\Gamma{^\alpha_{\beta\mu}}+\Gamma{^\alpha_{\beta\mu}}\Gamma{^\alpha_{\beta\nu}}-\Gamma{^\alpha_{\rho\mu}}\Gamma{^\alpha_{\beta\mu}}
```
That use Christoffel symbols who's depend of the metric components and have to be solved one by one (hardcoded atm because skill issue):

```math
\Gamma{^\alpha_{\beta\mu}} = \frac{1}{2} g^{\sigma\sigma}(\partial_\beta g_{\mu\rho}+\partial_\mu g_{\beta\rho}-\partial_\rho g_{\beta\mu})
```

Ricci tensor (Riemann contraction) :

```math
 R_{\mu\nu} = \nabla_\sigma \delta\Gamma{^\sigma_{\mu\nu}} - \nabla_\nu \delta\Gamma{^\sigma_{\nu\sigma}}
```
Einstein Tensor (or einstein field solution)
```math
   G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} Rg_{\mu\nu} = 0
```
