import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

here = Path(__file__).parent
d = np.load(here / "gaussian_hump.npy", allow_pickle=True).item()


Ns = np.asarray(d["N"], dtype=float)
errs = np.asarray(d["err_max"], dtype=float)
nI = d["nI"]
T = d["T"]

# reference line e^{-2N}, anchored at the first data point
ref = (errs[0] * 10e-2)*np.exp(-2 * (Ns-Ns[0]))

plt.semilogy(Ns, errs, "o-", label="Error")
plt.semilogy(Ns, ref, "--", label=r"$e^{-2N}$")
plt.xlabel("polynomial order $N$")
plt.ylabel(r"max error in $\zeta$ at south pole arrival [m]")
plt.title(f"Gaussian hump, $n_I$={nI}, t = {T/3600:.1f} h")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.savefig(here / "gaussian_hump_convergence.png", dpi=150)
plt.show()