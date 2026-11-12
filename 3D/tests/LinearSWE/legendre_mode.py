# Test 5: a single Legendre mode zeta0 = P_l(z/R), m0 = 0.
# Exact solution: zeta = P_l(z/R) cos(w_l t), w_l = sqrt(g D l(l+1)) / R.
# p-convergence on one mesh.

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as L

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from swe_sphere.geo import SphereGrid
from swe_sphere.dg import DG, LinearSWE

g = D = R = 1.0
l = 3
nI = 3
T = 1.0
Ns = range(2, 9)

Pl = L.Legendre.basis(l)
w = np.sqrt(g * D * l * (l + 1)) / R

err_max, err_L2 = [], []
for N in Ns:
    grid = SphereGrid(nI, N, R)
    solver = DG(grid, LinearSWE(g, D))
    mu = grid.x[2] / R

    q = np.zeros((4, grid.K, grid.Nq, grid.Nq))
    q[0] = Pl(mu)
    q = solver.run(q, T)

    e = q[0] - Pl(mu) * np.cos(w * T)
    err_max.append(np.abs(e).max())
    err_L2.append(np.sqrt(grid.integrate(e**2) / grid.integrate(Pl(mu)**2)))
    print(f"N={N}  max err={err_max[-1]:.2e}  rel L2 err={err_L2[-1]:.2e}")

# Save results. Load with: np.load("legendre_mode.npy", allow_pickle=True).item()
np.save(Path(__file__).with_name("legendre_mode.npy"), {
    "g": g, "D": D, "R": R, "l": l, "nI": nI, "K": grid.K, "T": T,
    "N": np.array(Ns), "err_max": np.array(err_max), "err_L2": np.array(err_L2),
})

fig, ax = plt.subplots(figsize=(5, 4))
ax.semilogy(Ns, err_max, "o-", label="max error")
ax.semilogy(Ns, err_L2, "s-", label="relative $L^2$ error")
ax.set_xlabel("polynomial order $N$")
ax.set_ylabel(f"error at $t={T}$")
ax.set_title(f"Legendre mode $\\ell={l}$, $n_I={nI}$, K={grid.K}")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(Path(__file__).with_name("legendre_mode.png"), dpi=150)
plt.show()
