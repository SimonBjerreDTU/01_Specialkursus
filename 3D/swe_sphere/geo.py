import numpy as np
from scipy.sparse import csr_matrix
from .lgl import LGL
from .mesh import icosahedral_mesh, merge_points

def cross_prod(a, b):
    return np.stack((
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]))
def dot_prod(a, b):
    return np.sum(a*b, axis=0)

class SphereGrid:
    def __init__(self, nI, N, R):
        self.nI, self.N, self.R = nI, N, R
        self.lgl = LGL(N)
        self.mesh = icosahedral_mesh(nI)
        self.K = self.mesh.K
        self.Nq = N + 1
        self.Np = self.Nq**2
        self.D = self.lgl.D

        self.vol_geo()
        self.edge_geo()
        self.uni_nodes()

    def vol_geo(self):
        r = self.lgl.r
        xi, eta = np.meshgrid(r, r, indexing="ij")
        v = self.mesh.VX[self.mesh.EToV]
        v = np.moveaxis(v, 2, 0)[..., None, None]
        v1, v2, v3, v4 = v[:, :, 0], v[:, :, 1], v[:, :, 2], v[:, :, 3]

        X = 1/4 * ((1-xi)*(1-eta)*v1 + (1+xi)*(1-eta)*v2 + (1+xi)*(1+eta)*v3 + (1-xi)*(1+eta)*v4)
        X_xi = 1/4 * ((1-eta)*(v2-v1) + (1+eta)*(v3-v4))
        X_eta = 1/4 * ((1-xi)*(v4-v1) + (1+xi)*(v3-v2))

        X_norm = np.sqrt(dot_prod(X,X))
        R = self.R
        self.n = X/X_norm
        self.x = R*self.n

        self.J = R**2/X_norm**3 * dot_prod(X, cross_prod(X_xi, X_eta))
        J_xi = R/X_norm**2 * cross_prod(X_eta, X)
        J_eta = R/X_norm**2 * cross_prod(X, X_xi)
        self.gxi = J_xi/self.J
        self.geta = J_eta/self.J

        w = self.lgl.w
        self.diag_mass = w[:, None] * w[None, :] *self.J

        self.lon = np.arctan2(self.x[1], self.x[0])
        self.lat = np.arcsin(np.clip(self.x[2]/R, -1, 1))

    def edge_geo(self):
        K, Nq, Np = self.K, self.Nq, self.Np
        N = self.N

        idx = np.arange(Nq)
        self.side_a = np.array([idx, np.full(Nq, N), idx, np.zeros(Nq, int)])
        self.side_c = np.array([np.zeros(Nq, int), idx, np.full(Nq, N), idx])
        local = self.side_a * Nq + self.side_c

        elem = np.arange(K)[:, None, None]
        self.vmapInterior = elem * Np + local[None]

        EToE, EToF = self.mesh.EToE, self.mesh.EToF
        self.vmapExterior = EToE[..., None] * Np + local[EToF]
        g = np.stack([
            -self.geta[:, :, :, 0], #side 0: eta=-1
            self.gxi[:, :, N, :], #side 1: xi=1
            self.geta[:, :, :, N], #side 2: eta=1
            -self.gxi[:, :, 0, :] #side 3: xi=-1
        ], axis=2)
        g_norm = np.sqrt(dot_prod(g,g))
        self.n_hat = g/g_norm
        self.lift = g_norm/self.lgl.w[0] #lift factor from Hesthaven&Warbuton

        n_edge = self.vmapInterior.size
        self.lift_matrix = csr_matrix((self.lift.ravel(), (self.vmapInterior.ravel(), np.arange(n_edge))), shape=(K*Np, n_edge))

    def uni_nodes(self):
        pts = self.x.reshape(3, -1).T/self.R
        unique_x, node_id = merge_points(pts, tol=1e-8)
        self.unique_x = unique_x*self.R
        self.node_id = node_id.reshape(self.K, self.Nq, self.Nq)

    # === Fields ===
    def cont_field(self, func):
        # Evaluate func(x), with x of shape (3, M)
        values = np.asarray(func(self.unique_x.T))
        return values[self.node_id]

    def trace(self, f):
        # Interior and exterior edge values of nodal fields:
        flat = f.reshape(f.shape[:-3]+(-1,))
        return flat[..., self.vmapInterior], flat[...,self.vmapExterior]

    def lift_edges(self, edge_vals):
        Nv = edge_vals.shape[0]
        out = (self.lift_matrix @ edge_vals.reshape(Nv, -1).T).T
        return out.reshape(Nv, self.K, self.Nq, self.Nq)

    # === Diff operators ===
    def d_xi(self, f):
        return self.D @ f

    def d_eta(self, f):
        return f @ self.D.T

    def grad(self, f):
        # Surface gradient
        return self.d_xi(f)*self.gxi + self.d_eta(f)*self.geta

    def div(self, F):
        #rowwise surface divergence of flux given columnwise, F[j] is column j of the flux
        gxi = self.gxi[:, None]
        geta = self.geta[:, None]
        return np.sum(self.d_xi(F)*gxi + self.d_eta(F)*geta, axis=0)

    def project(self, v):
        #project vector fields onto the tangent plane
        return v-self.n * dot_prod(self.n, v)

    def integrate(self, f):
        #lgl quadrature of nodal fields
        return np.sum(f*self.diag_mass, axis=(-3, -2, -1))