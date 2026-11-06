import numpy as np

from dataclasses import dataclass

@dataclass
class Mesh:
    VX: np.ndarray #(NV,3)
    EToV: np.ndarray #(K,4)
    EToE: np.ndarray #(K,4)
    EToF: np.ndarray #(K,4)

    @property
    def K(self):
        return self.EToV.shape[0]
    @property
    def NV(self):
        return self.VX.shape[0]

sideVertices = ((0, 1), (1, 2), (2, 3), (3, 0))

def icosahedron():
    phi = 0.5*(1+np.sqrt(5))
    verts = []
    for v1 in (-1, 1):
        for v2 in (-1, 1):
            verts += [(0, v1, v2*phi), (v1, v2*phi, 0), (v2*phi, 0, v1)]
    V = np.array(verts)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V

V = icosahedron()
print(V)

# Plotting made by Claude (Opus 5.5)
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=40, depthshade=False)

    # Unit sphere wireframe for reference
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 30), np.linspace(0, np.pi, 15))
    ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                      color="gray", alpha=0.2, linewidth=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Icosahedron vertices (NV = {V.shape[0]})")
    plt.show()