import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swe_sphere.geo import SphereGrid

R = 2.0
a = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
I_exact = 2*np.pi*R**2 * (np.e - 1/np.e)  # integral of exp(a.n) over the sphere

def error(nI, N):
    g = SphereGrid(nI, N, R)
    f = np.exp(np.tensordot(a, g.n, axes=1))
    return abs(g.integrate(f) - I_exact) / I_exact

fig, (ax_h, ax_p) = plt.subplots(1, 2, figsize=(10, 4))

nIs = np.array([1, 2, 4, 8])
for N in (2, 3, 4):
    ax_h.loglog(1/nIs, [error(nI, N) for nI in nIs], "o-", label=f"N={N}")
ax_h.set_xticks(1/nIs, [f"1/{n}" for n in nIs])
ax_h.minorticks_off()
ax_h.set(xlabel="h ~ 1/nI", ylabel="relative error", title="h-refinement")
ax_h.legend()
ax_h.grid(True, which="both", alpha=0.4)

Ns = np.arange(2, 13)
for nI in (2, 3, 4):
    ax_p.semilogy(Ns, [error(nI, N) for N in Ns], "o-", label=f"nI={nI}")
ax_p.set(xlabel="N", ylabel="relative error", title="p-refinement")
ax_p.legend()
ax_p.grid(True, which="both", alpha=0.4)

fig.tight_layout()
fig.savefig(Path(__file__).with_name("convergence_integrals.png"), dpi=150)
plt.show()
