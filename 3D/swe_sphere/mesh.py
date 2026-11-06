import numpy as np
from scipy.spatial import ConvexHull 

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
    F = ConvexHull(V).simplices.copy()

    # Make every face counter-clockwise seen from outside.
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    inward = np.einsum("ij,ij->i", np.cross(B - A, C - A), A + B + C) < 0
    F[inward] = F[inward][:, ::-1]
    return V, F

