# Test 6: a 10 m Gaussian hump at the north pole on an Earth-sized, 4 km deep ocean.
# The hump is axisymmetric, so the exact solution is a Legendre series
#   zeta(mu, t) = sum_l a_l P_l(mu) cos(w_l t),   w_l = sqrt(g D l(l+1)) / R,
# where mu = z/R and a_l are the Legendre coefficients of zeta0.
# The wave focuses at the south pole at about t = pi R / c.

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from swe_sphere.geo import SphereGrid
from swe_sphere.dg import DG, LinearSWE

R, g, D = 6.371e6, 9.81, 4000.0
A, sigma = 10.0, 800e3     # amplitude [m] and width [m]
nI = 4
N_plot = 6                 # order used for the propagation run
Ns = range(3, 7)           # orders for the p-convergence
n_frames = 100             # saved 3D frames in the propagation run (plus t = 0)
c = np.sqrt(g * D)
T = 40*3600          # arrival time at the south pole
L_max = 200                # terms in the exact series

def zeta0(mu):
    theta = np.arccos(np.clip(mu, -1, 1))
    return A * np.exp(-(R * theta)**2 / (2 * sigma**2))

def legendre_all(mu, L):
    # P_0..P_L at the points mu, by the three-term recurrence
    P = np.empty((L + 1,) + np.shape(mu))
    P[0] = 1.0
    P[1] = mu
    for l in range(1, L):
        P[l + 1] = ((2 * l + 1) * mu * P[l] - l * P[l - 1]) / (l + 1)
    return P

# Legendre coefficients of zeta0 by Gauss-Legendre quadrature
x_q, w_q = np.polynomial.legendre.leggauss(4 * L_max)
ls = np.arange(L_max + 1)
a = (2 * ls + 1) / 2 * (legendre_all(x_q, L_max) @ (w_q * zeta0(x_q)))
omega = np.sqrt(g * D * ls * (ls + 1)) / R

def exact(mu, t):
    P = legendre_all(mu, L_max)
    return np.tensordot(a * np.cos(omega * t), P, axes=1)

print(f"c = {c:.1f} m/s, T = {T/3600:.2f} h, last coefficient |a_L| = {abs(a[-1]):.1e}")

# --- p-convergence at t = T ---
errs = []
for N in Ns:
    grid = SphereGrid(nI, N, R)
    solver = DG(grid, LinearSWE(g, D))
    mu = grid.x[2] / R
    q = np.zeros((4, grid.K, grid.Nq, grid.Nq))
    q[0] = zeta0(mu)
    q = solver.run(q, T)
    errs.append(np.abs(q[0] - exact(mu, T)).max())
    print(f"N={N}  max err at T = {errs[-1]:.2e} m")

# --- propagation run for N = N_plot, saving the 3D field at every frame ---
grid = SphereGrid(nI, N_plot, R)
solver = DG(grid, LinearSWE(g, D))
mu = grid.x[2] / R

times = np.linspace(0, T, n_frames + 1)
q = np.zeros((4, grid.K, grid.Nq, grid.Nq))
q[0] = zeta0(mu)
zeta_num = [q[0].copy()]
for k in range(1, len(times)):
    q = solver.run(q, times[k] - times[k - 1])
    zeta_num.append(q[0].copy())
zeta_num = np.array(zeta_num)                          # (n_frames+1, K, Nq, Nq)
zeta_ex = np.array([exact(mu, t) for t in times])      # (n_frames+1, K, Nq, Nq)

# Save results. Load with: np.load("gaussian_hump.npy", allow_pickle=True).item()
np.save(Path(__file__).with_name("gaussian_hump.npy"), {
    "R": R, "g": g, "D": D, "A": A, "sigma": sigma, "nI": nI, "c": c, "T": T,
    # p-convergence at t = T
    "N": np.array(Ns), "err_max": np.array(errs),
    # propagation run for N = N_plot
    "N_plot": N_plot, "K": grid.K, "times": times,
    "x": grid.x,                 # (3, K, Nq, Nq) node coordinates [m]
    "EToV": grid.mesh.EToV, "VX": grid.mesh.VX,
    "zeta_num": zeta_num,        # (n_frames+1, K, Nq, Nq) DG solution [m]
    "zeta_exact": zeta_ex,       # (n_frames+1, K, Nq, Nq) exact solution at the nodes [m]
})

# --- 1D snapshots against the exact series ---
theta_deg = np.degrees(np.arccos(np.clip(mu, -1, 1))).ravel()
theta_fine = np.linspace(0, np.pi, 1000)
snap = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames]

fig, axes = plt.subplots(len(snap), 1, figsize=(8, 10), sharex=True)
for ax, k in zip(axes, snap):
    ax.plot(theta_deg, zeta_num[k].ravel(), ".", ms=1.5, color="C0", label="DG nodes")
    ax.plot(np.degrees(theta_fine), exact(np.cos(theta_fine), times[k]), "-", color="C3", lw=1, label="exact")
    ax.set_ylabel(r"$\zeta$ [m]")
    ax.set_title(f"t = {times[k]/3600:.1f} h", fontsize=10)
    ax.grid(alpha=0.3)
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("angular distance from north pole [deg]")
fig.suptitle(f"Gaussian hump, A={A:.0f} m, $\\sigma$={sigma/1e3:.0f} km, $n_I$={nI}, N={N_plot}, K={grid.K}")
fig.tight_layout()
fig.savefig(Path(__file__).with_name("gaussian_hump_snapshots.png"), dpi=150)

fig2, ax2 = plt.subplots(figsize=(5, 4))
ax2.semilogy(Ns, errs, "o-")
ax2.set_xlabel("polynomial order $N$")
ax2.set_ylabel(r"max error in $\zeta$ at south pole arrival [m]")
ax2.set_title(f"Gaussian hump, $n_I$={nI}, t = {T/3600:.1f} h")
ax2.grid(True, which="both", alpha=0.3)
fig2.tight_layout()
fig2.savefig(Path(__file__).with_name("gaussian_hump_convergence.png"), dpi=150)
plt.show()
