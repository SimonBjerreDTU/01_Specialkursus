import numpy as np

from dataclasses import dataclass

@dataclass
class Mesh:
    VX: np.ndarray #(NV,3)
    EToV: np.ndarray #(K,4)
    EToE: np.ndarray #(K,4)
    EToF: np.ndarray #(K,4)
    flag: np.ndarray #(K,4)

    @property
    def K(self):
        return self.EToV.shape[0]
    @property
    def NV(self):
        return self.VX.shape[0]

sideVertices = ((1, 2), (2, 3))