# ===
# A lot of this code is written by Claude Opus 5.5
# ===

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from PIL import Image

here = Path(__file__).parent
d = np.load(here / "gaussian_hump.npy", allow_pickle=True).item()
X = np.moveaxis(d["x"] / d["R"], 0, -1)            # unit sphere, (K, Nq, Nq, 3)
zeta = d["zeta_num"]                               # (n_frames, K, Nq, Nq)
err = np.abs(zeta - d["zeta_exact"])
emin, emax = 1e-4 * err.max(), err.max()

# zeta: white at 0, clear red/blue at +-1.5 m, almost black at +-10 m (beyond is clipped)
c = 0.5 * 1.5 / 10
zeta_cmap = LinearSegmentedColormap.from_list("hump", [
    (0, (0.03, 0.03, 0.12)), (0.5 - c, (0.15, 0.35, 0.85)), (0.5, "w"),
    (0.5 + c, (0.85, 0.1, 0.1)), (1, (0.12, 0.02, 0.02))])
panels = [  # (data, label, colour map, colour scale), fixed over all frames
    (zeta, r"$\zeta$ [m]", zeta_cmap, Normalize(-10, 10)),
    (np.maximum(err, emin), r"$|\zeta - \zeta_{exact}|$ [m]", plt.get_cmap("viridis"), LogNorm(emin, emax)),
]

def quads(f):
    # Corners of the (Nq-1)^2 small quads through the LGL nodes of every element
    return np.stack([f[:, :-1, :-1], f[:, 1:, :-1], f[:, 1:, 1:], f[:, :-1, 1:]], axis=3)

# Fixed camera a little off the initial hump (30N, 140E); only the facing side is drawn
elev, azim = 30 + 12, 140 - 20
e, a = np.radians([elev, azim])
cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
verts = quads(X).reshape(-1, 4, 3)
centres = verts.mean(axis=1)
mu = centres @ cam / np.linalg.norm(centres, axis=1)
vis = mu > 0
shade = (0.35 + 0.65 * mu[vis])[:, None]           # simple light from the camera

# Element boundaries (lifted slightly above the surface) and the visible poles
edges = np.concatenate([X[:, 0], X[:, -1], X[:, :, 0], X[:, :, -1]])
segs = np.stack([edges[:, :-1], edges[:, 1:]], axis=2).reshape(-1, 2, 3)
segs = 1.002 * segs[segs.mean(axis=1) @ cam > 0]
poles = np.array([[0, 0, 1.01], [0, 0, -1.01]])
poles = poles[poles @ cam > 0]

fig = plt.figure(figsize=(11, 5.5), dpi=120)
polys = []
for i, (_, label, cmap, norm) in enumerate(panels):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d", proj_type="ortho")
    ax.computed_zorder = False                     # draw in zorder
    polys.append(Poly3DCollection(verts[vis], linewidth=0.3, zorder=1))
    ax.add_collection3d(polys[-1])
    if i == 0:
        ax.add_collection3d(Line3DCollection(segs, colors="k", linewidths=0.4, alpha=0.3, zorder=2))
    ax.scatter(*poles.T, color="k", s=12, depthshade=False, zorder=3)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    ax.set_box_aspect((1, 1, 1), zoom=1.4)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=ax, shrink=0.6, pad=0.02, label=label)
fig.subplots_adjust(left=0, right=0.95, bottom=0.02, top=0.92, wspace=0.1)
title = (f"Linear SWE, Gaussian hump (A = {d['A']:.0f} m, $\\sigma$ = {d['sigma']/1e3:.0f} km), "
         f"N = {d['N_plot']}, K = {d['K']},  t = ")

# Render all frames, then save with one shared 256-colour palette from the first frame
# (a palette per frame, as PillowWriter does, makes the static colour bars flicker)
frames = []
for k, t in enumerate(d["times"]):
    for pc, (f, _, cmap, norm) in zip(polys, panels):
        rgba = cmap(norm(quads(f[k]).mean(axis=3).ravel()[vis]))
        rgba[:, :3] *= shade
        pc.set_facecolor(rgba)
        pc.set_edgecolor(rgba)                     # hides the seams between quads
    fig.suptitle(title + f"{t/3600:5.1f} h")
    fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
palette = frames[0].quantize(256)
frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
frames[0].save(here / "gaussian_hump.gif", save_all=True, append_images=frames[1:],
               duration=100, loop=0)
print("saved", here / "gaussian_hump.gif")
