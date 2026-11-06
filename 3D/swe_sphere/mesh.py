import numpy as np
from scipy.spatial import ConvexHull, cKDTree 

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

def connectivity(EToV):
    K = EToV.shape[0]
    side_owner = {}
    EToE = -np.ones((K, 4), dtype=np.int64)
    EToF = -np.ones((K, 4), dtype=np.int64)
    for e in range(K):
        for f, (s0, s1) in enumerate(sideVertices):
            start, end = EToV[e, s0], EToV[e, s1]
            key = (min(start, end), max(start, end))
            if key in side_owner:
                e2, f2 = side_owner.pop(key)
                EToE[e, f], EToF[e, f] = e2, f2
                EToE[e2, f2], EToF[e2, f2] = e, f
            else:
                side_owner[key] = (e, f)
    return EToE, EToF

def merge_points(P, tol):
    neighbours = cKDTree(P).query_ball_point(P, tol)
    first = np.array([min(nb) for nb in neighbours])
    keep, index = np.unique(first, return_inverse=True)
    return P[keep], index

def icosahedral_mesh(nI):
    V, F = icosahedron()
    quads = []
    for A, B, C in V[F]:
        def grid_point(a, b):
            return A + (a/nI) * (B - A) + (b/nI) * (C - A)

        triangles = []
        for a in range(nI):
            for b in range(nI - a):
                triangles.append((grid_point(a, b), grid_point(a + 1, b), grid_point(a, b + 1)))
                if a + b <= nI - 2:
                    triangles.append(
                        (grid_point(a + 1, b), grid_point(a + 1, b + 1), grid_point(a, b + 1))
                    )
        for A, B, C in triangles:
            a, b, c = 0.5 * (A + B), 0.5 * (B + C), 0.5 * (C + A)
            Centroid = (A + B + C) / 3.0
            quads += [(A, a, Centroid, c), (B, b, Centroid, a), (C, c, Centroid, b)]

    P = np.array(quads).reshape(-1, 3)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    VX, index = merge_points(P, tol=1e-10)
    EToV = index.reshape(-1, 4)

    EToE, EToF = connectivity(EToV)
    mesh = Mesh(VX=VX, EToV=EToV, EToE=EToE, EToF=EToF)

    return mesh

