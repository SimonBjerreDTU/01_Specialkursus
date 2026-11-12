import numpy as np
from .geo import SphereGrid

class LinearSWE:
    def __init__(self, g, D):
        self.g, self.D = g, D

    def flux(self, q):
        zeta, m = q[0], q[1:]
        F = np.zeros((3,) + q.shape)
        for j in range(3):
            F[j, 0] = m[j]
            F[j, 1 + j] = self.g * self.D * zeta
        return F

    def max_speed(self, q):
        return np.sqrt(self.g * self.D)

    def source(self, q):
        return 0.0


class DG:
    def __init__(self, grid, phys):
        self.grid = grid
        self.phys = phys

    def normal_flux(self, q):
        F = self.phys.flux(q)
        return np.sum(self.grid.n_hat[:, None] * F, axis=0)

    def rhs(self, q):
        g = self.grid
        qM, qP = g.trace(q)
        FnM, FnP = self.normal_flux(qM), self.normal_flux(qP)

        #FLUX
        c = np.maximum(self.phys.max_speed(qM), self.phys.max_speed(qP))
        Fstar = 0.5 * (FnM + FnP) - 0.5 * c * (qP - qM)

        dq = -g.div(self.phys.flux(q)) + g.lift_edges(FnM - Fstar) + self.phys.source(q)
        dq[1:] = g.project(dq[1:])
        return dq

    def dt(self, q, CFL=0.2):
        x = self.grid.x
        dx = min(np.linalg.norm(np.diff(x, axis=2), axis=0).min(), np.linalg.norm(np.diff(x, axis=3), axis=0).min())
        return CFL * dx / np.max(self.phys.max_speed(q))

    def step(self, q, dt):
        k1 = self.rhs(q)
        k2 = self.rhs(q + 0.5 * dt * k1)
        k3 = self.rhs(q + 0.5 * dt * k2)
        k4 = self.rhs(q + dt * k3)
        return q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def run(self, q, T, CFL=0.2):
        dt = self.dt(q, CFL)
        n = int(np.ceil(T / dt))
        dt = T / n
        for _ in range(n):
            q = self.step(q, dt)
        return q
