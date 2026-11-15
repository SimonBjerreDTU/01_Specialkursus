import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

here = Path(__file__).parent
d = np.load(here / "gaussian_hump_Hconvergence.npy", allow_pickle=True).item()

N = d["N"]
R = d["R"]
T = d["T"]
nIs = np.asarray(d["nI"], dtype=float)
errs = np.asarray(d["err_max"], dtype=float)

# mean element size: K = 60 nI^2 quads covering 4 pi R^2
Ks = 60 * nIs**2
hs = R * np.sqrt(4 * np.pi / Ks) / 1e3          # [km]

# reference slope h^(N+1), anchored at the finest grid
ref = errs[-1] * (hs / hs[-1])**(N + 1)

plt.loglog(hs, errs, "o-", label="Error")
plt.loglog(hs, ref, "--", label=rf"$h^{{{N + 1}}}$")
plt.xlabel(r"element size $h$ [km]")
plt.ylabel(r"max error in $\zeta$ at south pole arrival [m]")
plt.title(f"Gaussian hump, $N$={N}, t = {T/3600:.1f} h")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.savefig(here / "gaussian_hump_Hconvergence.png", dpi=150)
plt.show()
