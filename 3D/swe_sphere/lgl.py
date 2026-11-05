import numpy as np
from numpy.polynomial import legendre as npleg
from scipy.special import roots_jacobi

def nodes_and_weights(N):
    x, w = roots_jacobi(N-1, 1, 1)
    r = np.concatenate(([-1], np.sort(x), [1]))
    r = 0.5 * (r-r[::-1])
    PN = npleg.Legendre.basis(N)(r)
    w = 2.0/(N*(N+1)*PN**2)
    return r, w

def diff_matrix(r):
    diff = r[:, None] - r[None, :]
    np.fill_diagonal(diff, 1)
    bw = 1.0/np.prod(diff, axis=1)
    D = (bw[None, :]/bw[:, None]) /diff # There is a pattern in the diff matrix, with a lot of computational load. Optimization?
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(axis=1))
    return D

def vandermonde(r, N):
    V = np.empty((len(r), N+1))
    for a in range(N+1):
        V[:, a] = npleg.Legendre.basis(a)(r)*np.sqrt((2*a+1)/2.0)
    return V

class LGL:
    def __init__(self, N):
        self.N = N
        self.Nq = N + 1
        self.r, self.w = nodes_and_weights(N)
        self.D = diff_matrix(self.r)
        self.V = vandermonde(self.r, N)
        self.Vinv = np.linalg.inv(self.V)

        # For CFL time step
        dr = np.diff(self.r)
        self.dr = np.empty(self.Nq)
        self.dr[0] = dr[0]
        self.dr[-1] = dr[-1]
        self.dr[1:-1] = np.minimum(dr[:-1], dr[1:])

    def filter_matrix(self, alpha=30, s=10, nu=1):
        a = np.arange(self.Nq)
        sigma = np.exp(-alpha*(a/self.N)**s)
        F = self.V @ np.diag(sigma) @ self.Vinv
        return (1-nu) * np.eye(self.Nq) + nu*F
