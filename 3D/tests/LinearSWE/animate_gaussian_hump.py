# 3D animation of the Gaussian hump test from gaussian_hump.npy (run gaussian_hump.py first).
# Left: DG solution zeta. Right: |zeta_num - zeta_exact| on a log scale.
# The camera rotates around the sphere and moves from above the north pole to below
# the south pole, following the wave.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

here = Path(__file__).parent
d = np.load(here / "gaussian_hump.npy", allow_pickle=True).item()

x = d["x"] / d["R"]                     # unit sphere, (3, K, Nq, Nq)
times = d["times"]
zeta = d["zeta_num"]                    # (n_frames, K, Nq, Nq)
err = np.abs(zeta - d["zeta_exact"])

# (label, colour map, colour scale) for the two panels; values outside are clipped
panels = [
    (r"$\zeta$ [m]", plt.get_cmap("coolwarm"), Normalize(-1.5, 1.5)),
    (r"$|\zeta - \zeta_{exact}|$ [m]", plt.get_cmap("viridis"), LogNorm(1e-8, 1e-4)),
]

# Split every element into (Nq-1)^2 small quads through the LGL nodes
X = np.moveaxis(x, 0, -1)               # (K, Nq, Nq, 3)
verts = np.stack([X[:, :-1, :-1], X[:, 1:, :-1], X[:, 1:, 1:], X[:, :-1, 1:]], axis=3)
verts = verts.reshape(-1, 4, 3)
centres = verts.mean(axis=1)
centres /= np.linalg.norm(centres, axis=1, keepdims=True)

def face_values(f):
    # Mean of the four corners of every small quad, f: (K, Nq, Nq)
    return 0.25 * (f[:, :-1, :-1] + f[:, 1:, :-1] + f[:, 1:, 1:] + f[:, :-1, 1:]).ravel()

# plt.style.use("dark_background")
fig = plt.figure(figsize=(11, 5.5))
axes, polys = [], []
for i, (label, cmap, norm) in enumerate(panels):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    ax.set_proj_type("ortho")
    pc = Poly3DCollection(verts[:1], edgecolor="none")
    ax.add_collection3d(pc)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1), zoom=1.4)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, extend="both").set_label(label)
    axes.append(ax); polys.append(pc)
suptitle = fig.suptitle("")
fig.subplots_adjust(left=0, right=0.95, bottom=0.02, top=0.92, wspace=0.1)

def update(k):
    s = k / (len(times) - 1)
    elev, azim = 35 - 70 * s, -60 + 180 * s
    e, a = np.radians(elev), np.radians(azim)
    cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    mu = centres @ cam
    vis = mu > 0                                    # only the side facing the camera
    shade = (0.35 + 0.65 * mu[vis])[:, None]        # simple light from the camera
    for ax, pc, (_, cmap, norm), f in zip(axes, polys, panels, (zeta[k], err[k])):
        v = face_values(f)[vis]
        if isinstance(norm, LogNorm):
            v = np.maximum(v, 1e-30)                # log scale needs v > 0
        rgba = cmap(norm(v))
        rgba[:, :3] *= shade
        pc.set_verts(verts[vis])
        pc.set_facecolor(rgba)
        pc.set_edgecolor(rgba)                      # hides the seams between quads
        pc.set_linewidth(0.3)
        ax.view_init(elev=elev, azim=azim)
    suptitle.set_text(f"Linear SWE, Gaussian hump (A = {d['A']:.0f} m, $\\sigma$ = {d['sigma']/1e3:.0f} km), "
                      f"N = {d['N_plot']}, K = {d['K']},  t = {times[k]/3600:5.1f} h")
    return polys

anim = FuncAnimation(fig, update, frames=len(times), blit=False)
anim.save(here / "gaussian_hump.gif", writer=PillowWriter(fps=5), dpi=40)
print("saved", here / "gaussian_hump.gif")
