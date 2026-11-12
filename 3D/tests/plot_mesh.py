# Plot elements (curved edges) and LGL nodes of the SphereGrid on the sphere.
# Only the hemisphere facing the camera is drawn (matplotlib 3D has no occlusion).
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swe_sphere.geo import SphereGrid

nI, N, R = 3, 4, 1.0
elev, azim = 25, -60
g = SphereGrid(nI, N, R)

# Unit vector pointing towards the camera
e, a = np.radians(elev), np.radians(azim)
view = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
def front(P):  # P: (M,3)
    return P @ view > 0

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=elev, azim=azim)

# Light sphere surface
u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
ax.plot_surface(0.99*R*np.cos(u)*np.sin(v), 0.99*R*np.sin(u)*np.sin(v), 0.99*R*np.cos(v),
                color="lightgray", alpha=0.25, linewidth=0)

# Element edges: sample each side of the flat quad finely, project to the sphere, mask back side
t = np.linspace(0, 1, 30)[:, None]
for quad in g.mesh.EToV:
    V = g.mesh.VX[quad]
    for s in range(4):
        P = (1-t)*V[s] + t*V[(s+1) % 4]
        P = R*P/np.linalg.norm(P, axis=1, keepdims=True)
        P[~front(P)] = np.nan
        ax.plot(*P.T, color="k", lw=0.8)

# Nodes (unique LGL nodes, shared ones plotted once) and mesh vertices
X = g.unique_x
VX = R*g.mesh.VX/np.linalg.norm(g.mesh.VX, axis=1, keepdims=True)
ax.scatter(*X[front(X)].T, s=6, c="tab:blue", depthshade=False, label=f"LGL nodes ({len(X)} unique)")
# ax.scatter(*VX[front(VX)].T, s=25, c="tab:red", depthshade=False, label=f"mesh vertices ({len(VX)})")

ax.set_title(f"Quad mesh: nI={nI}, N={N}, K={g.K} elements")
# ax.set_box_aspect((1, 1, 1))
# ax.set_axis_off()
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(Path(__file__).with_name(f"mesh_nI{nI}_N{N}_K{g.K}.png"), dpi=150)
plt.show()
