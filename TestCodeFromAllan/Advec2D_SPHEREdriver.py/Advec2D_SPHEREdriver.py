# Advec2D_SPHEREdriver.py
#
# By Allan Peter Engsig-Karup (apek@dtu.dk), August 2025
#

import torch
import math
from types import SimpleNamespace
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import scipy.special
from scipy.special import roots_jacobi
from numpy.polynomial.legendre import leggauss
from scipy.special import gamma
import scipy.io
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time as timing

torch.set_num_threads(os.cpu_count())   # let Accelerate/BLAS use all cores
amp_dtype = None

# Mixed-precisin setup
MIXEDONOFF = True
if MIXEDONOFF:
    use_amp = True # False
    # Try bfloat16 first (usually more stable). If you’re on an older macOS/PyTorch combo, use float16.
    amp_dtype = torch.bfloat16  # or torch.float16
    amp_dtype = torch.float16  # or torch.float16
    amp_dtype = torch.float32  # or torch.float16
    autocast_ctx = torch.autocast(device_type="mps", dtype=amp_dtype, enabled=use_amp)

np.set_printoptions(precision=5, suppress=True, linewidth=200)
#torch.set_default_dtype(torch.float64)

# Use MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = "cpu" # for better precision use CPU

globals = SimpleNamespace()

# Polynomial order
globals.N = 4

# Modal filtering flag
globals.filterONOFF = False
storeGIFONOFF = False
PLOTFREQ = 20 

# Initial condition type
IC = 4

# Mesh type and generation level
generation = 3
mesh_type = "ico"

# Gaussian parameters
lambda0 = math.radians(225)
theta0 = math.radians(65)
sigma = math.radians(10)
v_phi = 0.4
v_lambda = 0.4

# --- Generate the mesh ---
def randomlyRotate(Pin):
    """
    Randomly rotates a set of 3D points.

    Parameters:
    - Pin: np.ndarray of shape (3, N), where each column is a 3D point.

    Returns:
    - Pout: np.ndarray of shape (3, N), rotated points.
    """
    _, _, R = np.linalg.svd(np.random.rand(3, 3))
    if np.linalg.det(R) < 0:
        R[:, [0, 1, 2]] = R[:, [2, 1, 0]]  # Flip for right-handed coordinate system
    Pout = R @ Pin.cpu().numpy()  # Ensure Pin is on CPU for numpy operations
    return Pout

def get_icosahedral_mesh():
    phi = (1 + math.sqrt(5)) / 2
    P = torch.tensor([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1]
    ], dtype=torch.float32).T

    P = P / torch.norm(P, dim=0, keepdim=True)
    P = P - torch.mean(P, dim=1, keepdim=True)
    P = randomlyRotate( P )

    # Basic icosahedron connectivity
    tri = torch.tensor([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7,10], [0,10,11],
        [1, 5, 9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1]
    ], dtype=torch.long)

    return P, tri

def refine_mesh(Pin, triin):
    """
    Refines a spherical triangle mesh by subdividing each triangle into 4 sub-triangles.

    Args:
        Pin: np.ndarray of shape (3, N), input vertex coordinates on the sphere.
        triin: np.ndarray of shape (T, 3), triangle vertex indices (1-based in MATLAB, 0-based here).

    Returns:
        P: np.ndarray, updated vertex coordinates.
        tri: np.ndarray, updated triangle indices (zero-based).
    """
    N = Pin.shape[1]

    # Construct edge lists from triangles
    edges_1 = np.vstack([triin[:, [0, 1]], triin[:, [1, 2]], triin[:, [2, 0]]])
    edges_2 = np.vstack([triin[:, [1, 2]], triin[:, [2, 0]], triin[:, [0, 1]]])
    edges_3 = np.vstack([triin[:, [2, 0]], triin[:, [0, 1]], triin[:, [1, 2]]])

    # Sort edge nodes to find unique edges
    edges_s1 = np.sort(edges_1, axis=1)
    edges_s2 = np.sort(edges_2, axis=1)
    edges_s3 = np.sort(edges_3, axis=1)

    # Find all unique edges
    edges_unique, inverse_indices = np.unique(edges_s1, axis=0, return_inverse=True)

    # Create new midpoints for each unique edge and normalize to unit sphere
    Pnew = Pin[:, edges_unique[:, 0]] + Pin[:, edges_unique[:, 1]]
    Pnew /= np.linalg.norm(Pnew, axis=0, keepdims=True)

    # Map edges to midpoint indices
    _, idx1 = np.unique(edges_s1, axis=0, return_inverse=True)
    _, idx2 = np.unique(edges_s2, axis=0, return_inverse=True)
    _, idx3 = np.unique(edges_s3, axis=0, return_inverse=True)

    # New triangles from each original triangle
    sz = triin.shape[0]
    tri = np.vstack([
        np.column_stack([edges_1[:sz, 0], N + idx1[:sz], N + idx3[:sz]]),
        np.column_stack([N + idx1[:sz], edges_2[:sz, 0], N + idx2[:sz]]),
        np.column_stack([N + idx3[:sz], N + idx2[:sz], edges_3[:sz, 0]]),
        np.column_stack([N + idx1[:sz], N + idx2[:sz], N + idx3[:sz]])
    ])

    # Combine old and new vertices
    P = np.hstack([Pin, Pnew])

    return P, tri

def refine_mesh_old(P, tri):
    N = P.shape[1]
    edges = torch.cat([
        tri[:, [0,1]],
        tri[:, [1,2]],
        tri[:, [2,0]]
    ], dim=0)

    edges_sorted, _ = torch.sort(edges, dim=1)
    edges_unique, inverse_indices = torch.unique(edges_sorted, dim=0, return_inverse=True)

    # Midpoints
    Pnew = P[:, edges_unique[:, 0]] + P[:, edges_unique[:, 1]]
    Pnew = Pnew / torch.norm(Pnew, dim=0, keepdim=True)

    P = torch.cat([P, Pnew], dim=1)
    mid_idx = torch.arange(N, P.shape[1])

    idx1 = inverse_indices[:tri.shape[0]]
    idx2 = inverse_indices[tri.shape[0]:2*tri.shape[0]]
    idx3 = inverse_indices[2*tri.shape[0]:]

    m1 = N + idx1
    m2 = N + idx2
    m3 = N + idx3

    t0 = torch.stack([tri[:,0], m1, m3], dim=1)
    t1 = torch.stack([tri[:,1], m2, m1], dim=1)
    t2 = torch.stack([tri[:,2], m3, m2], dim=1)
    t3 = torch.stack([m1, m2, m3], dim=1)

    tri = torch.cat([t0, t1, t2, t3], dim=0)
    return P, tri

def generate_sphere_mesh(generation, mesh_type="ico"):
    if mesh_type == "ico":
        P, tri = get_icosahedral_mesh()
        #print("P=", P.shape)
        #print("P=", P[:,:10])
        #assert(1==0)  # Debugging point to check mesh generation
    else:
        raise ValueError(f"Unsupported mesh type: {mesh_type}")

    for _ in range(generation):
        P, tri = refine_mesh(P, tri)
    return P, tri

def loadmesh(filename):
    """
    Load a mesh from a MATLAB .mat file containing:
      - P: 3 x Np array of node coordinates
      - EToV: K x 3 array of triangle vertex indices (1-based)

    Returns:
      - P: numpy array of shape (3, Np)
      - tri: numpy array of shape (K, 3) with 0-based indexing
    """
    mat = scipy.io.loadmat(filename)
    
    # Extract and ensure proper shape
    if 'P' not in mat or 'EToV' not in mat:
        raise KeyError("MAT file must contain 'P' and 'EToV' variables.")
    
    P = mat['P']
    tri = mat['EToV']

    # Flatten MATLAB's column-major format if needed
    if P.shape[0] == 1 or P.shape[1] == 1:
        P = P.reshape((3, -1))

    # Convert to 0-based indexing for Python
    tri = tri.astype(np.int64) - 1

    return P, tri

filename = 'fixedmesh.mat'
if 1:
    P, tri = generate_sphere_mesh(generation, mesh_type)
else:
    print(f"Loading mesh from {filename}")
    P, tri = loadmesh(filename)
    print("Mesh loaded successfully!")
    
EToV = tri
VX, VY, VZ = P[0], P[1], P[2]
K = EToV.shape[0]

print("P=", P.shape)
#print("P=", P[:,:10])
print("EToV: max:", np.max(EToV), "min:", np.min(EToV))
#assert(1==0)  # Debugging point to check mesh generation

# === Correct Triangle Orientations ===
num_flipped = 0
for k in range(EToV.shape[0]):
    v1 = P[:, EToV[k, 0]]
    v2 = P[:, EToV[k, 1]]
    v3 = P[:, EToV[k, 2]]
    e1 = v2 - v1
    e2 = v3 - v1
    # Ensure e1 and e2 are PyTorch tensors (not NumPy)
    e1 = torch.tensor(e1, dtype=torch.float32, device=device) if isinstance(e1, np.ndarray) else e1
    e2 = torch.tensor(e2, dtype=torch.float32, device=device) if isinstance(e2, np.ndarray) else e2

    n_tri = torch.cross(e1, e2, dim=0)
    centroid = (v1 + v2 + v3) / 3
    centroid = torch.tensor(centroid, dtype=torch.float32, device=device) if isinstance(centroid, np.ndarray) else centroid

    if torch.dot(n_tri, centroid) < 0:
        EToV[k, [1, 2]] = EToV[k, [2, 1]]
        num_flipped += 1

if num_flipped > 0:
    raise RuntimeError(f"Corrected {num_flipped} triangles with incorrect orientation.")

# === Initialize Reference Element ===
def jacobi_gl(alpha, beta, N):
    if N == 1:
        return torch.tensor([-1.0, 1.0], dtype=torch.float32, device=device)
    xint, _ = jacobi_gq(alpha+1, beta+1, N-2)
    return torch.tensor(np.concatenate([[-1.0], xint, [1.0]]), dtype=torch.float32, device=device)

def jacobi_gq(alpha, beta, N):
    """
    Stable Jacobi Gauss quadrature for given alpha, beta, and order N.
    Returns: torch.tensor x (nodes), w (weights)
    """

    alpha = float(alpha)
    beta = float(beta)

    if N == 0:
        x = torch.tensor([-(alpha - beta) / (alpha + beta + 2)], dtype=torch.float32)
        w = torch.tensor([2.0], dtype=torch.float32)
        return x, w

    J = np.zeros((N + 1, N + 1), dtype=np.float64)

    h1 = 2 * np.arange(N + 1) + alpha + beta
    # Diagonal entries
    J_diag = -0.5 * (alpha**2 - beta**2) / ((h1 + 2) * h1)
    if np.any(np.abs(h1) < 10 * np.finfo(float).eps):        
        #print("h1=", h1,"alpha=", alpha, "beta=", beta)
        #print(J_diag)
        # for each value that is nan, set it to zero
        J_diag[np.isnan(J_diag)] = 0.0
    np.fill_diagonal(J, J_diag)

    # Off-diagonal entries (sub-diagonal, upper filled by symmetry)
    n = np.arange(1, N + 1)
    hn = h1[:-1]  # h1(1:N) in MATLAB

    numerator = n * (n + alpha + beta) * (n + alpha) * (n + beta)
    denominator = (hn + 1) * (hn + 3) 
    offdiag = 2.0 / (hn + 2) * np.sqrt(numerator / denominator)

    # Assign sub-diagonal
    J[np.arange(1, N + 1), np.arange(N)] = offdiag

    # Symmetrize
    J = J + J.T - np.diag(np.diag(J))

    # Special fix for very small alpha + beta
    if abs(alpha + beta) < 10 * np.finfo(float).eps:
        J[0, 0] = 0.0

    try:
        eigvals, eigvecs = np.linalg.eigh(J)
    except np.linalg.LinAlgError as e:
        raise RuntimeError("JacobiGQ eigenvalue computation failed: " + str(e))

    x = torch.tensor(eigvals, dtype=torch.float32)
    v0 = eigvecs[0, :]
    const = 2**(alpha + beta + 1) / (alpha + beta + 1)
    weight_factor = const * gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 1)
    w = torch.tensor(v0**2 * weight_factor, dtype=torch.float32)
    return x, w

def warp_factor(N, rout):
    LGLr = jacobi_gl(0, 0, N)
    req = torch.linspace(-1, 1, N+1, device=device)

    Veq = vandermonde1D(N, req)
    Nr = rout.numel()
    Pmat = torch.stack([JacobiP(rout, 0, 0, i) for i in range(N+1)], dim=0)
    Lmat = torch.linalg.solve(Veq.T, Pmat)
    warp = (Lmat.T @ (LGLr - req)).flatten()
    zerof = (rout.abs() < 1.0 - 1.0e-10)
    sf = 1.0 - (zerof * rout)**2
    warp = warp / sf + warp * (zerof.float() - 1)
    return warp

def Nodes2D(N):
    alpopt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
              1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]

    alpha = alpopt[N] if N < 16 else 5/3

    Np = (N+1)*(N+2)//2
    L1 = torch.zeros(Np, device=device)
    L3 = torch.zeros(Np, device=device)

    sk = 0
    for n in range(1, N+2):
        for m in range(1, N+3-n):
            L1[sk] = (n - 1)/N
            L3[sk] = (m - 1)/N
            sk += 1
    L2 = 1.0 - L1 - L3
    x = -L2 + L3
    y = (-L2 - L3 + 2*L1)/math.sqrt(3.0)

    blend1 = 4*L2*L3
    blend2 = 4*L1*L3
    blend3 = 4*L1*L2

    warpf1 = warp_factor(N, L3 - L2)
    warpf2 = warp_factor(N, L1 - L3)
    warpf3 = warp_factor(N, L2 - L1)

    warp1 = blend1 * warpf1 * (1 + (alpha * L1)**2)
    warp2 = blend2 * warpf2 * (1 + (alpha * L2)**2)
    warp3 = blend3 * warpf3 * (1 + (alpha * L3)**2)
    
    x = x + 1.0*warp1 + math.cos(2*math.pi/3)*warp2 + math.cos(4*math.pi/3)*warp3
    y = y + 0.0*warp1 + math.sin(2*math.pi/3)*warp2 + math.sin(4*math.pi/3)*warp3

    return x, y

def JacobiP(x, alpha, beta, N):
    x = x.to(device)  # Ensure x is on the correct device    
    #alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
    #beta  = torch.tensor(beta, dtype=torch.float32, device=device)

    x = x.view(-1)
    PL = torch.zeros((x.shape[0], N+1), dtype=torch.float32, device=device)
    gamma0 = 2**(alpha + beta + 1) / (alpha + beta + 1) \
        *gamma(alpha+1)*gamma(beta+1) / gamma(alpha + beta + 1)  
    #gamma0b = torch.lgamma(alpha + 1.0)*torch.lgamma(beta + 1.0)
    #gamma0c = torch.lgamma(alpha + beta + 1.0)
    #gamma0 = gamma0*gamma0b.exp()/gamma0c.exp()
    PL[:, 0] = 1.0 / np.sqrt(gamma0)

    if N == 0:
        return PL[:, 0]

    gamma1 = (alpha + 1)*(beta + 1)/(alpha + beta + 3)*gamma0
    PL[:, 1] = ((alpha + beta + 2)*x/2 + (alpha - beta)/2)/np.sqrt(gamma1)

    aold = 2.0 / (2 + alpha + beta) * np.sqrt((alpha + 1)*(beta + 1)/(alpha + beta + 3))
    for n in range(1, N):
        h1 = 2 * n + alpha + beta
        anew = 2.0 / (h1 + 2) * np.sqrt(
            (n+1)*(n+1+alpha+beta)*(n+1+alpha)*(n+1+beta)/((h1+1)*(h1+3))
        )
        bnew = -(alpha**2 - beta**2) / (h1 * (h1 + 2))
        PL[:, n+1] = (1.0 / anew) * (
            -(aold * PL[:, n-1]) + (x - bnew)*PL[:, n]
        )
        aold = anew
    return PL[:, N]

def vandermonde1D(N, r):
    """
    Initialize the 1D Vandermonde Matrix, V_{ij} = P_j(r_i),
    where P_j is the j-th Legendre polynomial evaluated at node r_i.
    """
    V1D = torch.zeros((r.numel(), N + 1), dtype=torch.float32, device=device)
    for j in range(N + 1):
        V1D[:, j] = JacobiP(r, 0, 0, j)
    return V1D.to(device)

def xytors(x, y):
    sqrt3 = torch.sqrt(torch.tensor(3.0))
    L1 = (sqrt3*y + 1.0)/3.0
    L2 = (-3.0*x - sqrt3*y + 2.0)/6.0
    L3 = (3.0*x - sqrt3*y + 2.0)/6.0
    r = -L2 + L3 - L1
    s = -L2 - L3 + L1
    return r.to(device), s.to(device)

N = globals.N

x_ref, y_ref = Nodes2D(N)
#print('x_ref=',x_ref)
#print()
#print('y_ref=',y_ref)
#assert(1==0)
r, s = xytors(x_ref, y_ref)
#print()
#print('r=', r)
#print()
#print('s=', s)
#assert(1==0)
Np = (N + 1)*(N + 2)//2
Nfp = N + 1
Nfaces = 3

globals.Np = Np
globals.Nfp = Nfp
globals.Nfaces = Nfaces
globals.r = r
globals.s = s
globals.NODETOL = 1e-12 # CAN BE VERY SENSITIVE TO THIS CHOICE

# === Vandermonde Matrix on Simplex ===
def rstoab(r, s):
    a = torch.where(
        s != 1.0,
        2.0 * (1 + r) / (1 - s) - 1,
        -torch.ones_like(s)
    )
    b = s
    return a, b
#print()
#print('rstoab(r,s)=', rstoab(r, s))
#print(JacobiP(r,1,1,1)) # OK
#print()
#print(JacobiP(r,1,1,2)) #OK
#print()
#print(JacobiP(r,1,1,3))
#assert(1==0)

def Simplex2DP(a, b, i, j):
    h1 = JacobiP(a, 0, 0, i)
    h2 = JacobiP(b, 2*i + 1, 0, j)
    #print('***\nh1=', h1, '\nh2=', h2, '\ni=', i, 'j=', j)  # Debugging output
    return torch.sqrt(torch.tensor(2.0)) * h1 * h2 * (1 - b)**i

def Vandermonde2D(N, r, s):
    a, b = rstoab(r, s)
    V2D = torch.zeros((len(r), (N+1)*(N+2)//2), device=device)
    sk = 0
    for i in range(N+1):
        for j in range(N+1 - i):
            V2D[:, sk] = Simplex2DP(a, b, i, j)
            sk += 1
    return V2D

# === Build V and MassMatrix ===
V = Vandermonde2D(N, r, s)
#print('V=', V.cpu().numpy())
#print()
#i = 1
#j = 0
#a,b = rstoab(r, s)
#print("h2=",JacobiP(b,2*i+1,0,j))
#assert(1==0)
invV = torch.inverse(V)
MassMatrix = invV.T @ invV

def Dmatrices2D(N, r, s, V):
    Vr, Vs = GradVandermonde2D(N, r, s)
    # print()
    # print('Vr=', Vr.cpu().numpy())
    # print()
    # print('Vs=', Vs.cpu().numpy())
    # #assert(1==0)
    invV = torch.inverse(V)
    Dr = Vr @ invV
    Ds = Vs @ invV
    return Dr, Ds

def GradSimplex2DP(a, b, id, jd):
    # Ensure a, b are torch tensors
    a = torch.tensor(a, dtype=torch.float32) if not torch.is_tensor(a) else a
    b = torch.tensor(b, dtype=torch.float32) if not torch.is_tensor(b) else b

    fa  = JacobiP(a, 0, 0, id)
    dfa = GradJacobiP(a, 0, 0, id)
    gb  = JacobiP(b, 2 * id + 1, 0, jd)
    dgb = GradJacobiP(b, 2 * id + 1, 0, jd)

    dmodedr = dfa * gb
    if id > 0:
        dmodedr *= (0.5 * (1.0 - b)) ** (id - 1)

    dmodeds = dfa * gb * (0.5 * (1.0 + a))
    if id > 0:
        dmodeds *= (0.5 * (1.0 - b)) ** (id - 1)

    tmp = dgb * (0.5 * (1.0 - b)) ** id
    if id > 0:
        tmp = tmp - 0.5 * id * gb * (0.5 * (1.0 - b)) ** (id - 1)

    dmodeds = dmodeds + fa * tmp

    scale = 2.0 ** (id + 0.5)
    dmodedr *= scale
    dmodeds *= scale

    return dmodedr, dmodeds


def GradVandermonde2D(N, r, s):
    a, b = rstoab(r, s)
    Np = (N + 1) * (N + 2) // 2
    V2Dr = torch.zeros((len(r), Np), device=device)
    V2Ds = torch.zeros_like(V2Dr)

    sk = 0
    for i in range(N + 1):
        for j in range(N + 1 - i):
            V2Dr[:, sk], V2Ds[:, sk] = GradSimplex2DP(a, b, i, j)
            sk += 1
    return V2Dr, V2Ds

def GradJacobiP(x, alpha, beta, N):
    if N == 0:
        return torch.zeros_like(x)
    return torch.sqrt(torch.tensor(N * (N + alpha + beta + 1))) * JacobiP(x, alpha + 1, beta + 1, N - 1)

Dr, Ds = Dmatrices2D(N, r, s, V)
# print('Dr=',Dr.cpu().numpy())
# print('Ds=',Ds.cpu().numpy())
# print()
# print(GradJacobiP(r, 1, 2, N))
# print(JacobiP(r, 1+1, 2+1, N-1))
# print()
# print(GradJacobiP(r, 1, 1, N))
# print(JacobiP(r, 1+1, 1+1, N-1))
# print()
# print(GradJacobiP(r, 1, 0, N))
# print(JacobiP(r, 1+1, 0+1, N-1))
# assert(1==0)

va = EToV[:, 0]  # shape: (K,)
vb = EToV[:, 1]
vc = EToV[:, 2]

#VX = VX.to(device)
#VY = VY.to(device)

#device = torch.device('mps')  # or 'cuda' or 'cpu' depending on your setup

# Convert VX and indexing arrays to tensors on the correct device
VX = torch.tensor(VX, dtype=torch.float32, device=device) if isinstance(VX, np.ndarray) else VX.to(device)
VY = torch.tensor(VY, dtype=torch.float32, device=device) if isinstance(VY, np.ndarray) else VY.to(device)

va = torch.tensor(va, dtype=torch.long, device=device) if isinstance(va, np.ndarray) else va.to(device)
vb = torch.tensor(vb, dtype=torch.long, device=device) if isinstance(vb, np.ndarray) else vb.to(device)
vc = torch.tensor(vc, dtype=torch.long, device=device) if isinstance(vc, np.ndarray) else vc.to(device)

r = r.to(device)
s = s.to(device)

# Shape: (Np, K)
x = 0.5 * (-(r[:, None] + s[:, None]) * VX[va] + (1 + r[:, None]) * VX[vb] + (1 + s[:, None]) * VX[vc])
y = 0.5 * (-(r[:, None] + s[:, None]) * VY[va] + (1 + r[:, None]) * VY[vb] + (1 + s[:, None]) * VY[vc])

NODETOL = globals.NODETOL

#fmask1 = torch.nonzero(torch.abs(s + 1) < NODETOL, as_tuple=True)[0]
#fmask2 = torch.nonzero(torch.abs(r + s) < NODETOL, as_tuple=True)[0]
#fmask3 = torch.nonzero(torch.abs(r + 1) < NODETOL, as_tuple=True)[0]
#
#Fmask = torch.stack([fmask1, fmask2, fmask3], dim=1)  # shape: (Nfp, 3)

def build_Fmask(N):
    # Total number of nodes
    Np = (N + 1)*(N + 2)//2

    # Create a map from node index to (i,j) barycentric coords
    sk = 0
    ids = torch.zeros((N+1, N+1), dtype=torch.long) - 1  # initialize with -1
    for i in range(N+1):
        for j in range(N+1 - i):
            ids[i, j] = sk
            sk += 1

    # Face 1: s = -1 => i+j=N
    fmask2 = torch.tensor([ids[i, N - i].item() for i in range(N+1)], dtype=torch.long)

    # Face 2: r+s = 0 => j=0
    fmask3 = torch.tensor([ids[i, 0].item() for i in range(N+1)], dtype=torch.long)

    # Face 3: r = -1 => i=0
    fmask1 = torch.tensor([ids[0, j].item() for j in range(N+1)], dtype=torch.long)
    #print("fmaskX=")
    #print(fmask1, fmask2, fmask3)
    return torch.stack([fmask1, fmask2, fmask3], dim=1).to(device)

Fmask = build_Fmask(N)

Fx = x[Fmask.T.reshape(-1), :]
Fy = y[Fmask.T.reshape(-1), :]

if 0:
    print("Fmask shape = ", Fmask.shape)
    print("Fmask = ", Fmask.cpu().numpy())
    print("Fx shape = ", Fx.shape)
    print("Fx = ", Fx.cpu().numpy()[:,:5])
    print("Fy shape = ", Fy.shape)
    print("Fy = ", Fy.cpu().numpy()[:,:5])    

    print("V shape = ", V.shape)
    print("V = ", V.cpu().numpy())
    print("V^T = ", V.cpu().numpy().T)

    print("r=",r)
    print("s=",s)

def Lift2D():
    Emat = torch.zeros((Np, Nfaces * Nfp), device=device)
    # face1
    faceR = r[Fmask[:, 0]]
    V1D = vandermonde1D(N, faceR)
    massEdge1 = torch.linalg.inv(V1D @ V1D.T)
    Emat[Fmask[:, 0], :Nfp] = massEdge1
    # face2
    faceR = r[Fmask[:, 1]]
    V1D = vandermonde1D(N, faceR)
    massEdge2 = torch.linalg.inv(V1D @ V1D.T)
    Emat[Fmask[:, 1],Nfp:2*Nfp] = massEdge2
    # face3
    faceS = s[Fmask[:, 2]]
    V1D = vandermonde1D(N, faceS)
    massEdge3 = torch.linalg.inv(V1D @ V1D.T)
    Emat[Fmask[:, 2],2*Nfp:3*Nfp] = massEdge3
    #print("Emat = \n", Emat.cpu().numpy())
    return V @ (V.T @ Emat)

def Lift2Dnotworking():
    row_idx = Fmask.T.reshape(-1)  # (Nfaces * Nfp,)
    col_idx = torch.arange(Nfaces * Nfp, device=device)
    Emat = torch.zeros((Np, Nfaces * Nfp), device=device)
    Emat[row_idx, col_idx] = 1.0
    #print("Emat = \n", Emat.cpu().numpy())
    return V @ (V.T @ Emat)

LIFT = Lift2D()

#print()
#
#print("LIFT shape = ", LIFT.shape)
#print("LIFT = ", LIFT[:,:].cpu().numpy())
#
#assert(1==0)  # Debugging point to check Lift2D

def GeometricFactors2D(x, y, Dr, Ds):
    xr = Dr @ x
    xs = Ds @ x
    yr = Dr @ y
    ys = Ds @ y

    J = xr * ys - xs * yr
    rx =  ys / J
    sx = -yr / J
    ry = -xs / J
    sy =  xr / J

    return rx, sx, ry, sy, J

rx, sx, ry, sy, J = GeometricFactors2D(x, y, Dr, Ds)


# === Compute Differentiation Matrices ===
def GradJacobiP(x, alpha, beta, N):
    if N == 0:
        return torch.zeros_like(x)
    return torch.sqrt(torch.tensor(N*(N + alpha + beta + 1.0))) * \
        JacobiP(x, alpha + 1, beta + 1, N - 1)

def GradSimplex2DP(a, b, id, jd):
    fa = JacobiP(a, 0, 0, id)
    dfa = GradJacobiP(a, 0, 0, id)
    gb = JacobiP(b, 2*id + 1, 0, jd)
    dgb = GradJacobiP(b, 2*id + 1, 0, jd)

    dmodedr = dfa * gb
    if id > 0:
        dmodedr *= (0.5 * (1 - b))**(id - 1)

    dmodeds = dfa * gb * (0.5 * (1 + a))
    if id > 0:
        dmodeds *= (0.5 * (1 - b))**(id - 1)

    tmp = dgb * (0.5 * (1 - b))**id
    if id > 0:
        tmp -= 0.5 * id * gb * (0.5 * (1 - b))**(id - 1)
    dmodeds += fa * tmp

    scale = 2**(id + 0.5)
    return scale * dmodedr, scale * dmodeds

def GradVandermonde2D(N, r, s):
    a, b = rstoab(r, s)
    V2Dr = torch.zeros((len(r), (N+1)*(N+2)//2), device=device)
    V2Ds = torch.zeros_like(V2Dr)
    sk = 0
    for i in range(N+1):
        for j in range(N+1 - i):
            V2Dr[:, sk], V2Ds[:, sk] = GradSimplex2DP(a, b, i, j)
            sk += 1
    return V2Dr, V2Ds

Vr, Vs = GradVandermonde2D(N, r, s)
Dr = Vr @ torch.inverse(V)
Ds = Vs @ torch.inverse(V)

MassMatrix2D = MassMatrix
invM2D = torch.inverse(MassMatrix2D)
Dr2D = Dr
Ds2D = Ds

globals.V = V
globals.invV = invV
globals.MassMatrix = MassMatrix
globals.Dr = Dr
globals.Ds = Ds
globals.Dr2D = Dr2D
globals.Ds2D = Ds2D
globals.MassMatrix2D = MassMatrix2D
globals.invM2D = invM2D

# === Modal Filter Setup ===

if globals.filterONOFF:

    Nc = 4  # filter cutoff mode
    sp = 2  # power of filter

    def ConformingVandermonde2D_Ordering(N, r, s):
        a, b = rstoab(r, s)
        Np = (N + 1) * (N + 2) // 2
        V = torch.zeros((len(r), Np), device=device)
        O = torch.zeros(Np, dtype=torch.int32, device=device)
        ModeOrder = torch.zeros((Np, 2), dtype=torch.int32, device=device)

        sk = 0
        # Vertex A
        V[:, sk] = (1 - a) * (1 - b) / 4
        O[sk] = 1
        ModeOrder[sk] = torch.tensor([1, 1])
        sk += 1

        # Vertex B
        V[:, sk] = (1 + a) * (1 - b) / 4
        O[sk] = 1
        ModeOrder[sk] = torch.tensor([1, 1])
        sk += 1

        # Vertex C
        V[:, sk] = (1 + b) / 2
        O[sk] = 1
        ModeOrder[sk] = torch.tensor([0, 1])
        sk += 1

        if N == 1:
            return V, O, ModeOrder

        # Edge AB
        for i in range(1, N):
            V[:, sk] = 0.25 * (1 - a) * (1 + a) * ((0.5 * (1 - b)) ** (i + 1)) * JacobiP(a, 1, 1, i - 1)
            O[sk] = 1 + i
            ModeOrder[sk] = torch.tensor([2 + (i - 1), i + 1])
            sk += 1

        # Edge BC
        for j in range(1, N):
            V[:, sk] = ((1 + a) * (1 - b) * (1 + b) * JacobiP(b, 1, 1, j - 1)) / 8
            O[sk] = 1 + j
            ModeOrder[sk] = torch.tensor([1, 2 + (j - 1)])
            sk += 1

        # Edge CA
        for j in range(1, N):
            V[:, sk] = ((1 - a) * (1 - b) * (1 + b) * JacobiP(b, 1, 1, j - 1)) / 8
            O[sk] = 1 + j
            ModeOrder[sk] = torch.tensor([1, 2 + (j - 1)])
            sk += 1

        if N == 2:
            return V, O, ModeOrder

        # Interior modes
        for i in range(1, N):
            for j in range(1, N - i):
                V[:, sk] = (
                    0.125 * (1 - a) * (1 + a) * ((0.5 * (1 - b)) ** (i + 1)) * (1 + b)
                    * JacobiP(a, 1, 1, i - 1) * JacobiP(b, 2 * i + 1, 1, j - 1)
                )
                O[sk] = 1 + i + j
                ModeOrder[sk] = torch.tensor([2 + (i - 1), (i + 1) + 1 + (j - 1)])
                sk += 1

        return V, O, ModeOrder

    def ConformingFilterFunction2DClaes(N, O, Nc, sp):
        Np = (N + 1) * (N + 2) // 2
        Fdiag = torch.ones(Np, dtype=torch.float32, device=device)

        # Allan: mild filter
        ids = (O == N).nonzero(as_tuple=True)[0]
        Fdiag[ids] = 0.95
        ids = (O == N - 1).nonzero(as_tuple=True)[0]
        Fdiag[ids] = 0.99

        return Fdiag

    VDM2D, Ordering, ModeOrder = ConformingVandermonde2D_Ordering(N, globals.r, globals.s)
    Fdiag = ConformingFilterFunction2DClaes(N, Ordering, Nc, sp)

    # Construct filter matrix
    Filter2D = VDM2D @ (torch.diag(Fdiag) @ torch.inverse(VDM2D))
    globals.Filter2D = Filter2D

    print("Filter for FS stabilization created.")


# === Overintegration Setup ===
Nfine = 3 * N

# Fine mesh nodal set
xfine, yfine = Nodes2D(Nfine)
rfine, sfine = xytors(xfine, yfine)

# Fine Vandermonde and Mass Matrix
Vfine = Vandermonde2D(Nfine, rfine, sfine)
invVfine = torch.inverse(Vfine)
MassMatrix2Dfine = invVfine.T @ invVfine

# Coarse-to-fine Vandermonde
Vc2f = Vandermonde2D(N, rfine, sfine)
Ic2f2D = Vc2f @ torch.inverse(V)  # coarse to fine interpolation
IIf = Ic2f2D @ torch.eye(Np, device=device)

# Fine differentiation matrices
Vr_fine, Vs_fine = GradVandermonde2D(Nfine, rfine, sfine)
Dr2Df = Vr_fine @ torch.inverse(Vfine)
Ds2Df = Vs_fine @ torch.inverse(Vfine)

# === Geometric and Metric Arrays ===
total_area = 0.0
rx2D = torch.zeros((Np, K), device=device)
ry2D = torch.zeros_like(rx2D)
rz2D = torch.zeros_like(rx2D)
sx2D = torch.zeros_like(rx2D)
sy2D = torch.zeros_like(rx2D)
sz2D = torch.zeros_like(rx2D)
J2D = torch.zeros_like(rx2D)

x3D = torch.zeros_like(rx2D)
y3D = torch.zeros_like(rx2D)
z3D = torch.zeros_like(rx2D)

for k in range(K):
    v1 = P[:, EToV[k, 0]]
    v2 = P[:, EToV[k, 1]]
    v3 = P[:, EToV[k, 2]]
    #print("k=", k, "EToV(k,:)=",EToV[k,:], "v1=", v1, "v2=", v2, "v3=", v3)
    #if k==3:
    #    assert(1==0)
    # Map from reference to physical
    xk = 0.5 * (-(r + s) * v1[0] + (1 + r) * v2[0] + (1 + s) * v3[0])
    yk = 0.5 * (-(r + s) * v1[1] + (1 + r) * v2[1] + (1 + s) * v3[1])
    zk = 0.5 * (-(r + s) * v1[2] + (1 + r) * v2[2] + (1 + s) * v3[2])

    # Project to sphere
    norms = torch.sqrt(xk**2 + yk**2 + zk**2)
    xk /= norms
    yk /= norms
    zk /= norms

    x3D[:, k] = xk
    y3D[:, k] = yk
    z3D[:, k] = zk

    # Geometric factors
    xr = Dr @ xk
    xs = Ds @ xk
    yr = Dr @ yk
    ys = Ds @ yk
    zr = Dr @ zk
    zs = Ds @ zk

    xr_vec = torch.stack([xr, yr, zr], dim=1)
    xs_vec = torch.stack([xs, ys, zs], dim=1)
    n_vec = torch.cross(xr_vec, xs_vec, dim=1)
    J = torch.sqrt(torch.sum(n_vec**2, dim=1))
    n_unit = n_vec / J[:, None]

    ar = torch.cross(xs_vec, n_unit, dim=1) / J[:, None]
    as_ = torch.cross(n_unit, xr_vec, dim=1) / J[:, None]

    rx2D[:, k] = ar[:, 0]
    ry2D[:, k] = ar[:, 1]
    rz2D[:, k] = ar[:, 2]
    sx2D[:, k] = as_[:, 0]
    sy2D[:, k] = as_[:, 1]
    sz2D[:, k] = as_[:, 2]

    J2D[:, k] = J

    # Scaled element mass matrix
    M_k = torch.diag(J) @ MassMatrix
    total_area += M_k.sum().item()

# print("P=",P.shape)
# print("r=", r.cpu().numpy())
# print("s=", s.cpu().numpy())
# print("x3D=", x3D[:,:5].cpu().numpy())
# print("y3D=", y3D[:,:5].cpu().numpy())
# print("z3D=", z3D[:,:5].cpu().numpy())
# print("J2D=", J2D[:,:5].cpu().numpy())
# assert(1==0)

Jinv  = 1.0 / J2D  
globals.Jinv = Jinv


# === Normals for spherical surface triangles ===
def Normals3Dspheretriangles():
    fx = x3D[Fmask.reshape(-1), :]
    fy = y3D[Fmask.reshape(-1), :]
    fz = z3D[Fmask.reshape(-1), :]

    nx = torch.zeros((3 * Nfp, K), device=device)
    ny = torch.zeros_like(nx)
    nz = torch.zeros_like(nx)
    sJ = torch.zeros_like(nx)

    ref_normals = torch.tensor([[0, -1], [1, 1], [-1, 0]], dtype=torch.float32, device=device)

    xr = Dr @ x3D
    yr = Dr @ y3D
    zr = Dr @ z3D
    xs = Ds @ x3D
    ys = Ds @ y3D
    zs = Ds @ z3D

    J2 = (yr*zs - zr*ys)**2 + (zr*xs - xr*zs)**2 + (xr*ys - yr*xs)**2

    rx = ((xs**2 + ys**2 + zs**2) * xr - (xr*xs + yr*ys + zr*zs) * xs) / J2
    ry = ((xs**2 + ys**2 + zs**2) * yr - (xr*xs + yr*ys + zr*zs) * ys) / J2
    rz = ((xs**2 + ys**2 + zs**2) * zr - (xr*xs + yr*ys + zr*zs) * zs) / J2

    sx = ((xr**2 + yr**2 + zr**2) * xs - (xr*xs + yr*ys + zr*zs) * xr) / J2
    sy = ((xr**2 + yr**2 + zr**2) * ys - (xr*xs + yr*ys + zr*zs) * yr) / J2
    sz = ((xr**2 + yr**2 + zr**2) * zs - (xr*xs + yr*ys + zr*zs) * zr) / J2

    for f in range(3):
        nr, ns = ref_normals[f]
        ids = slice(f * Nfp, (f + 1) * Nfp)
        idf = Fmask[:, f]
        for k in range(K):
            nx[ids, k] = nr * rx[idf, k] + ns * sx[idf, k]
            ny[ids, k] = nr * ry[idf, k] + ns * sy[idf, k]
            nz[ids, k] = nr * rz[idf, k] + ns * sz[idf, k]
            sJ[ids, k] = torch.sqrt(nx[ids, k]**2 + ny[ids, k]**2 + nz[ids, k]**2)
            nx[ids, k] /= sJ[ids, k]
            ny[ids, k] /= sJ[ids, k]
            nz[ids, k] /= sJ[ids, k]
    return nx, ny, nz, sJ

nx, ny, nz, sJ_faces = Normals3Dspheretriangles()

print("Assumption: unit sphere. Normals computed.")

# === Surface Jacobians for Curved Faces ===

def compute_surface_jacobians_curved(x3D, y3D, z3D, Fmask, Nfp):
    Nfaces = 3
    K = x3D.shape[1]
    r1D, w1D = jacobi_gq(0.0, 0.0, Nfp - 1)
    M1D = torch.diag(w1D).to(device)  # ensures it's on the same device as ds
    one = torch.ones(Nfp, device=device)

    V1D = vandermonde1D(Nfp - 1, r1D)
    Dr1D = grad_vandermonde1D(Nfp - 1, r1D) @ torch.linalg.inv(V1D)

    sJ = torch.zeros((Nfp * Nfaces, K), device=device)

    for k in range(K):
        for f in range(Nfaces):
            ids = Fmask[:, f]
            xf = x3D[ids, k]
            yf = y3D[ids, k]
            zf = z3D[ids, k]

            dxf = Dr1D @ xf
            dyf = Dr1D @ yf
            dzf = Dr1D @ zf

            ds = torch.sqrt(dxf**2 + dyf**2 + dzf**2)
            L = torch.sum(M1D @ ds)
            sj_face = (L / (one @ M1D @ one)) * one

            idg = slice(f * Nfp, (f + 1) * Nfp)
            sJ[idg, k] = sj_face

    return sJ

def grad_vandermonde1D(N, r):
    """
    Compute gradient of the 1D Vandermonde matrix at points r.
    Each column corresponds to the derivative of the j-th Legendre polynomial.
    """
    dV1D = torch.zeros((r.numel(), N + 1), dtype=torch.float32, device=device)
    for j in range(N + 1):
        dV1D[:, j] = GradJacobiP(r, 0, 0, j)
    return dV1D

sJ = compute_surface_jacobians_curved(x3D, y3D, z3D, Fmask, Nfp)
# Fscale = sJ / J2D[Fmask, :]

# Fmask: shape (Nfp, Nfaces)
# We need to gather rows from J2D of shape (Np, K)

Fscale = torch.zeros_like(sJ)
for f in range(Nfaces):
    ids = slice(f * Nfp, (f + 1) * Nfp)
    Fscale[ids, :] = sJ[ids, :] / J2D[Fmask[:, f], :]

if 0:
    print("Fscale=", Fscale.cpu().numpy()[:,:5])
    print("Fmask=", Fmask.cpu().numpy())
    print("J2D=", J2D.cpu().numpy()[:,:5])
    print("sJ=", sJ.cpu().numpy()[:,:5])
    print("Fscale=", Fscale.cpu().numpy()[:,:5])
    #assert(1==0)

#Reshape_fortran function found here: https://stackoverflow.com/questions/63960352/reshaping-order-in-pytorch-fortran-like-index-ordering
def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

# Build connectivity matrix
def tiConnect2D(EToV):
    Nfaces = 3
    K = EToV.shape[0]
    Nnodes = EToV.max().item()

    if isinstance(EToV, np.ndarray):
        EToV = torch.tensor(EToV, dtype=torch.long, device=device)

    # Create list of all faces (each row = [node1, node2] for one face)
    fnodes = []
    for i in range(Nfaces):
        i1 = i
        i2 = (i + 1) % Nfaces
        fnodes.append(EToV[:, [i1, i2]])
    fnodes = torch.cat(fnodes, dim=0)  # shape [3K, 2]
    fnodes, _ = torch.sort(fnodes, dim=1)
    #print("fnodes=", fnodes.cpu().numpy()[:10])
    # Initialize connectivity
    EToE = torch.arange(K, device=device).unsqueeze(1).repeat(1, Nfaces)  # (K, 3)
    EToF = torch.arange(Nfaces, device=device).repeat(K, 1)               # (K, 3)
    #print("EToE=", EToE.cpu().numpy()[:30])
    #print("EToF=", EToF.cpu().numpy()[:30])    

    # Unique edge IDs for face-matching
    id = fnodes[:, 0] * (Nnodes + 1) + fnodes[:, 1]  # shape [3K]

    spNodeToNode = torch.stack([
        id,
        torch.arange(Nfaces * K, device=device),
        EToE.T.reshape(-1),
        EToF.T.reshape(-1)
    ], dim=1)  # shape [3K, 4]
    #print("spNodeToNode=", spNodeToNode.cpu().numpy()[:10])
    # Sort by edge ID (first column), then by face index
    sorted_spNodeToNode = spNodeToNode[spNodeToNode[:, 0].argsort()]
    #print("sorted_spNodeToNode=", sorted_spNodeToNode.shape)

    # Find matches (adjacent rows with same edge ID)
    id1 = sorted_spNodeToNode[:-1, 0]
    id2 = sorted_spNodeToNode[1:, 0]
    matches = torch.where(id1 == id2)[0]

    # matchL and matchR: each row contains [id, face_idx, EToE, EToF]
    # matchL = sorted_spNodeToNode[matches]
    # matchR = sorted_spNodeToNode[matches + 1]

    # Build matchL and matchR as in MATLAB
    matchL = torch.cat([sorted_spNodeToNode[matches], sorted_spNodeToNode[matches + 1]], dim=0)
    matchR = torch.cat([sorted_spNodeToNode[matches + 1], sorted_spNodeToNode[matches]], dim=0)

    #print("mathcL=", matchL.shape, "matchR=", matchR.shape)
    #print("matchL=", matchL.cpu().numpy()[:30])
    #print("matchR=", matchR.cpu().numpy()[:30])

    # Save to a .mat file
    # Make sure matchL and matchR are on CPU and converted to numpy
    #matchL_np = matchL.cpu().numpy()
    #matchR_np = matchR.cpu().numpy()
    #scipy.io.savemat('match_data.mat', {
    #    'matchL': matchL_np,
    #    'matchR': matchR_np
    #})
    #print("Stored the match data vectors in 'match_data.mat'.")

    # Flattened EToE and EToF column-wise
    EToE_flat = EToE.T.reshape(-1)
    EToF_flat = EToF.T.reshape(-1)

    # Assign matches
    EToE_flat[matchL[:, 1]] = matchR[:, 2]
    EToF_flat[matchL[:, 1]] = matchR[:, 3]
    #print("EToE_flat=", EToE_flat.cpu().numpy()[:30])
    #print("EToF_flat=", EToF_flat.cpu().numpy()[:30])    
    # Reshape back to (K, Nfaces)
    EToE = EToE_flat.reshape(Nfaces, K).T
    EToF = EToF_flat.reshape(Nfaces, K).T

    return EToE, EToF

def tiConnect2D_almostworking(EToV):

    Nfaces = 3
    K = EToV.shape[0]
    Nnodes = EToV.max().item()

    if isinstance(EToV, np.ndarray):
        EToV = torch.tensor(EToV, dtype=torch.long)    

    #K = EToV.shape[0]
    #Nnodes = torch.max(EToV)
    #Nfaces = 3

    # create list of all faces
    fnodes = []
    for i in range(0, Nfaces):
        if i == Nfaces - 1:
            fnodes.append(EToV[:, torch.tensor([i, 0])])
        else:
            fnodes.append(EToV[:, torch.tensor([i, i + 1])])

    fnodes = torch.cat(fnodes, dim=0)

    fnodes, index = torch.sort(fnodes)
    fnodes = fnodes  #- 1

    # set up default element to element and Element to faces connectivity
    #EToE = torch.arange(0, K, device=device).unsqueeze(0).T * torch.LongTensor(
    #    1, Nfaces).fill_(1).to(device)
    #EToF = torch.LongTensor(K, 1).fill_(1).to(device) * torch.arange(
    #    0, Nfaces, device=device)

    # Ensure all tensors are created on the same device
    EToE = torch.arange(K, device=device).unsqueeze(1).repeat(1, Nfaces)  # shape (K, Nfaces)
    EToF = torch.arange(Nfaces, device=device).repeat(K, 1)               # shape (K, Nfaces)

    # Unique face ID
    id = fnodes[:, 0] * (Nnodes + 1) + fnodes[:, 1]

    # Build spNodeToNode on same device
    spNodeToNode = torch.cat([
        id.unsqueeze(1),                                # [3K, 1]
        torch.arange(Nfaces * K, device=device).unsqueeze(1),  # [3K, 1]
        EToE.T.reshape(-1, 1),                           # [3K, 1]
        EToF.T.reshape(-1, 1)                            # [3K, 1]
    ], dim=1)


    #id = fnodes[:, 0] * (Nnodes + 1) + fnodes[:, 1]
    #spNodeToNode = torch.cat([
    #    id.unsqueeze(0).T,
    #    torch.arange(0, Nfaces * K, device=device).unsqueeze(0).T,
    #    EToE.T.flatten().unsqueeze(0).T,
    #    EToF.T.flatten().unsqueeze(0).T
    #], dim=1)

    # # Now we sort by global face number.
    sp = spNodeToNode.cpu().detach().numpy()
    ss = torch.tensor(sorted(sp, key=lambda t: (t[0], t[1])),
                      dtype=torch.long,
                      device=device)
    # # find matches in the sorted face list
    indices = torch.where(ss[0:-1, 0] == ss[1:, 0])[0]

    # # make links reflexive
    matchL = torch.cat([ss[indices, :], ss[indices + 1, :]])
    matchR = torch.cat([ss[indices + 1, :], ss[indices, :]])

    # # insert matches
    Ef = EToE.T.flatten()
    Ef[matchL[:, 1]] = matchR[:, 2]  #.type(torch.cuda.DoubleTensor)
    EToE = reshape_fortran(Ef, EToF.shape)

    Ef = EToF.T.flatten()
    Ef[matchL[:, 1]] = matchR[:, 3]  #.type(torch.cuda.DoubleTensor)
    EToF = reshape_fortran(Ef, EToE.shape)
    return EToF, EToE


def tiConnect2Dnotworking(EToV):
    """
    Triangle face connectivity builder for triangular meshes.

    Args:
        EToV (torch.Tensor): (K x 3) tensor of element-to-vertex connectivity (1-based indexing)

    Returns:
        EToE (torch.Tensor): (K x 3) element-to-element connectivity
        EToF (torch.Tensor): (K x 3) element-to-face connectivity
    """
    Nfaces = 3
    K = EToV.shape[0]
    Nnodes = EToV.max().item()

    if isinstance(EToV, np.ndarray):
        EToV = torch.tensor(EToV, dtype=torch.long)    

    # Face node pairs
    fnodes = torch.cat([
        EToV[:, [0, 1]],
        EToV[:, [1, 2]],
        EToV[:, [2, 0]]
    ], dim=0)

    fnodes = torch.sort(fnodes, dim=1)[0] - 1  # convert to 0-based indexing
    print("fnodes=", fnodes.shape)
    # Build unique face IDs
    face_ids = fnodes[:, 0] * Nnodes + fnodes[:, 1]
    face_numbers = torch.arange(Nfaces * K)
    elements = (face_numbers // Nfaces).long()
    local_faces = (face_numbers % Nfaces).long()
    print("elements=",elements[:10])
    print("local_faces=",local_faces[:10])

    # Stack into a sortable array
    face_data = torch.stack([face_ids, face_numbers, elements, local_faces], dim=1)
    face_data_np = face_data.numpy()

    # Sort by face IDs
    sorted_faces = face_data_np[np.argsort(face_data_np[:, 0])]

    # Find duplicates (shared faces)
    matches = np.where(sorted_faces[1:, 0] == sorted_faces[:-1, 0])[0]

    matchL = torch.tensor(sorted_faces[matches], dtype=torch.long)
    matchR = torch.tensor(sorted_faces[matches + 1], dtype=torch.long)

    # Initialize self-references
    EToE = torch.arange(K).unsqueeze(1).repeat(1, Nfaces)
    EToF = torch.arange(Nfaces).unsqueeze(0).repeat(K, 1)

    # Flatten for assignment
    EToE_flat = EToE.view(-1)
    EToF_flat = EToF.view(-1)

    # Assign neighbors
    EToE_flat[matchL[:, 1]] = matchR[:, 2]
    EToF_flat[matchL[:, 1]] = matchR[:, 3]

    # Reshape back
    EToE = EToE_flat.view(Nfaces,K)
    EToF = EToF_flat.view(Nfaces,K)

    return EToE.T, EToF.T


# Build element-to-element and element-to-face connectivity
EToE, EToF = tiConnect2D(EToV)
print("EToV=", EToV[:10,:])
print("EToE=", EToE.cpu().numpy()[:10,:])
print("EToF=", EToF.cpu().numpy()[:,:])
#assert(1==0)

def BuildMaps2Dsphere(K, Np, Nfp, Nfaces, EToE, EToF, EToV, Fmask, x3D, y3D, z3D, VX, VY, VZ, NODETOL, device=None, dtype=None):
    """
    Purpose: Connectivity and boundary tables in the K # of Np elements
    
    Parameters:
    K : int
        Number of elements
    Np : int
        Number of nodes per element
    Nfp : int
        Number of face nodes
    Nfaces : int
        Number of faces per element
    EToE : torch.Tensor
        Element to element connectivity (0-indexed)
    EToF : torch.Tensor
        Element to face connectivity (0-indexed)
    EToV : torch.Tensor
        Element to vertex connectivity (0-indexed)
    Fmask : torch.Tensor
        Face mask (0-indexed)
    x3D, y3D, z3D : torch.Tensor
        3D coordinates (0-indexed)
    VX, VY, VZ : torch.Tensor
        Vertex coordinates (0-indexed)
    NODETOL : float
        Node tolerance
    device : torch.device
        Device for tensors
    dtype  : torch.dtype
        Precision type to be used on chosen device

    Returns:
    mapM, mapP, vmapM, vmapP, vmapB, mapB : torch.Tensor
        Connectivity maps
    """
    # ---- Defaults: CPU + float64 unless explicitly overridden ----
    # --- defaults and normalization ---
    if isinstance(device, str):
        device = torch.device(device)
    if dtype is None:
        dtype = torch.float64
    if device.type == "mps" and dtype == torch.float64:
        # choose one of these:
        # dtype = torch.float32             # allow MPS (lower precision)
        # or:
        device = torch.device("cpu")      # keep float64 precision on CPU
        # or:
        # raise TypeError("MPS does not support float64")
        pass

    #if device is None:
    #    device = torch.device("cpu")
    #elif isinstance(device, str):
    #    device = torch.device(device)
    #
    #if dtype is None:
    #    dtype = torch.float64  # keep math in double on CPU

    # --- helpers that accept numpy or torch and place on device with right dtype ---
    def _to_long(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.long)
        return torch.as_tensor(x, device=device, dtype=torch.long)

    def _to_float(x):
        # Decide a safe target for float tensors
        # If someone ever passes device='mps' together with float64, keep floats on CPU.
        target_device = device
        if isinstance(device, torch.device):
            is_mps = (device.type == "mps")
        else:
            is_mps = (str(device) == "mps")

        if is_mps and dtype == torch.float64:
            target_device = torch.device("cpu")

        # Make a tensor without accidentally landing on MPS with float64
        if isinstance(x, torch.Tensor):
            t = x
        else:
            # For NumPy inputs, create directly on a safe device
            init_device = target_device if not (is_mps and dtype == torch.float64) else torch.device("cpu")
            t = torch.as_tensor(x, device=init_device)

        # 1) move device first
        if t.device != target_device:
            t = t.to(device=target_device)

        # 2) then cast dtype
        if t.dtype != dtype:
            t = t.to(dtype=dtype)
        return t

    # ---- Ensure tensor dtypes/devices (keep indices long; floats as requested dtype) ----
    # index tensors
    EToE  = _to_long(EToE)
    EToF  = _to_long(EToF)
    EToV  = _to_long(EToV)
    Fmask = _to_long(Fmask)

    # coordinate tensors
    x3D = _to_float(x3D)
    y3D = _to_float(y3D)
    z3D = _to_float(z3D)
    VX  = _to_float(VX)
    VY  = _to_float(VY)
    VZ  = _to_float(VZ)
    
    NODETOL = torch.as_tensor(NODETOL, device=device, dtype=dtype)

    # Number volume nodes consecutively (0-indexed in Python)
    # MATLAB: nodeids = reshape(1:K*Np, Np, K) creates column-major
    # PyTorch equivalent with proper indexing conversion:
    # Transpose for column-major equivalent    
    nodeids = torch.arange(K * Np, device=device, dtype=torch.long).reshape((K, Np)).T

    # Initialize arrays
    vmapM = torch.zeros((Nfp, Nfaces, K), dtype=torch.long, device=device)
    vmapP = torch.zeros((Nfp, Nfaces, K), dtype=torch.long, device=device)
    
    # MATLAB: mapM = (1:K*Nfp*Nfaces)' creates column vector starting from 1
    # Python: 0-indexed
    #mapM = torch.arange(K * Nfp * Nfaces, device=device)
    #mapP = torch.zeros_like(mapM, dtype=torch.long, device=device)
    mapM = torch.arange(K * Nfp * Nfaces, device=device, dtype=torch.long)
    mapP = torch.zeros_like(mapM, dtype=torch.long, device=device).reshape((Nfp, Nfaces, K))
    #mapP = mapP.reshape((Nfp, Nfaces, K))  # Reshape to match MATLAB ordering
    
    # Find index of face nodes with respect to volume node ordering
    for k1 in range(K):  # 0-indexed loop
        for f1 in range(Nfaces):  # 0-indexed loop
            # MATLAB: vmapM(:,f1,k1) = nodeids(Fmask(:,f1), k1)
            # Fmask should be converted to 0-indexed before passing to this function
            vmapM[:, f1, k1] = nodeids[Fmask[:, f1], k1]
    
    # MATLAB: one = ones(1, Nfp) creates row vector
    one = torch.ones((1, Nfp), device=device, dtype=dtype)
    
    tmpx3D = x3D.T.reshape(-1)
    tmpy3D = y3D.T.reshape(-1)
    tmpz3D = z3D.T.reshape(-1)

    for k1 in range(K):  # 0-indexed loop
        for f1 in range(Nfaces):  # 0-indexed loop
            # Find neighbor (EToE and EToF should be 0-indexed before passing)
            k2 = EToE[k1, f1].item()  # Convert to Python int
            f2 = EToF[k1, f1].item()  # Convert to Python int
            
            # Reference length of edge
            # EToV should be 0-indexed before passing
            v1 = EToV[k1, f1].item()  # Convert to Python int
            v2 = EToV[k1, (f1 + 1) % Nfaces].item()  # Convert to Python int
            
            refd = torch.sqrt((VX[v1] - VX[v2])**2 + (VY[v1] - VY[v2])**2 + (VZ[v1] - VZ[v2])**2)
            
            # Find volume node numbers of left and right nodes
            vidM = vmapM[:, f1, k1]
            vidP = vmapM[:, f2, k2]
            
            x1 = tmpx3D[vidM]  # Shape: (Nfp,)
            y1 = tmpy3D[vidM]  # Shape: (Nfp,)
            z1 = tmpz3D[vidM]  # Shape: (Nfp,)
            x2 = tmpx3D[vidP]  # Shape: (Nfp,)
            y2 = tmpy3D[vidP]  # Shape: (Nfp,)
            z2 = tmpz3D[vidP]  # Shape: (Nfp,)
            
            #print("k=", k1, "f1=", f1, "k2=", k2, "f2=", f2)
            #print("vidM=", vidM.cpu().numpy()[:5], "vidP=", vidP.cpu().numpy()[:5])
            #print("x1=",x1.cpu().numpy())
            #print("y1=",y1.cpu().numpy())
            #print("z1=",z1.cpu().numpy())
            #print("x2=",x2.cpu().numpy())
            #print("y2=",y2.cpu().numpy())
            #print("z2=",z2.cpu().numpy())

            # Debug: Print shapes to understand the issue
            #print(f"x1.shape: {x1.shape}, one.shape: {one.shape}")
            #print(f"vidM.shape: {vidM.shape}, vidP.shape: {vidP.shape}")
            #print(f"Nfp: {Nfp}")

            # MATLAB broadcasting: x1 is column vector (Nfp,1), one is row vector (1,Nfp)
            # x1*one creates (Nfp, Nfp) matrix where each row is x1
            x1 = x1.unsqueeze(1) * one  # (Nfp,1) * (1,Nfp) -> (Nfp,Nfp)
            y1 = y1.unsqueeze(1) * one  # (Nfp,1) * (1,Nfp) -> (Nfp,Nfp)
            z1 = z1.unsqueeze(1) * one  # (Nfp,1) * (1,Nfp) -> (Nfp,Nfp)
            x2 = x2.unsqueeze(1) * one  # (Nfp,1) * (1,Nfp) -> (Nfp,Nfp)
            y2 = y2.unsqueeze(1) * one  # (Nfp,1) * (1,Nfp) -> (Nfp,Nfp)
            z2 = z2.unsqueeze(1) * one  # (Nfp,1) * (1,Nfp) -> (Nfp,Nfp)
            
            # Compute distance matrix
            # MATLAB: D = (x1 -x2').^2 + (y1-y2').^2 + (z1-z2').^2
            D = (x1 - x2.T)**2 + (y1 - y2.T)**2 + (z1 - z2.T)**2

            # Debug: Print the distance matrix
            #print("D=", D.cpu().numpy())

            # Find matching nodes
            # MATLAB: [idM, idP] = find(sqrt(abs(D))<NODETOL*refd)
            mask = torch.abs(D) < NODETOL * refd**2
            idM, idP = torch.where(mask)
            #print("idM=", idM.cpu().numpy(), "idP=", idP.cpu().numpy())
            # idM, idP = torch.where(mask)
            count = idM.numel()  # safer than len(idM)

            if count < Nfp:
                # Optional: tiny bit of debug info
                dmin = torch.sqrt(torch.abs(D)).min().item()
                raise RuntimeError(
                    f"[BuildMaps2Dsphere] Incomplete face match: found {count} < Nfp={Nfp} "
                    f"for element {k1}, face {f1} (neighbor {k2}, face {f2}). "
                    f"NODETOL={float(NODETOL)}; min distance on this pair ~ {dmin:.3e}. "
                    "Consider increasing NODETOL or checking EToE/EToF/Fmask."
                )                

            #print("vidM[idM]=", vidM[idM].cpu().numpy())
            #print("vidP[idP]=", vidP[idP].cpu().numpy())

            # command to allow user to press any key to continue...
            #input("Press any key to continue...")  

            if len(idM) > 0:  # Only update if there are matches
                vmapP[idM, f1, k1] = vidP[idP]
                # MATLAB: mapP(idM,f1,k1) = idP + (f2-1)*Nfp+(k2-1)*Nfaces*Nfp
                # Convert to 0-indexed: idP + f2*Nfp + k2*Nfaces*Nfp
                mapP[idM, f1, k1] = idP + f2 * Nfp + k2 * Nfaces * Nfp
        
    # Reshape vmapM and vmapP to be vectors and create boundary node list
    # MATLAB uses column-major flattening, PyTorch uses row-major by default
    # We need to permute dimensions to match MATLAB's column-major ordering before flattening
    vmapP = vmapP.permute(2, 1, 0).flatten()  # Permute to get column-major order, then flatten
    vmapM = vmapM.permute(2, 1, 0).flatten()
    mapP  = mapP.permute(2, 1, 0).flatten()

    #print("vmapM=", vmapM.cpu().numpy()[:20])
    #print("vmapP=", vmapP.cpu().numpy()[:20])
    #print("mapM=", mapM.cpu().numpy()[:20])
    #print("mapP=", mapP.cpu().numpy()[:20])

    # Find boundary nodes (where vmapP == vmapM indicates boundary)
    # MATLAB: mapB = find(vmapP==vmapM)
    mapB = torch.where(vmapP == vmapM)[0]
    vmapB = vmapM[mapB]

    #print("vmapB=", vmapB.cpu().numpy()[:20])
    #print("mapB=", mapB.cpu().numpy()[:20])
    
    #assert(1==0)
    return mapM, mapP, vmapM, vmapP, vmapB, mapB

# Build connectivity maps - in high precision! dtype = float64!
mapM, mapP, vmapM, vmapP, vmapB, mapB = BuildMaps2Dsphere(K, Np, Nfp, Nfaces, EToE, EToF, EToV, Fmask, x3D, y3D, z3D, VX, VY, VZ, 1e-8, device="cpu", dtype=None)

# === Check total area ===
print('Check mass matrix and jacobians by integrating at the sphere surface')
print(f'Exact area is 4*pi : {4 * math.pi:.8f}')
print(f'Computed total area: {total_area:.8f}')

# === Initial condition: test function or Gaussian ===
if IC == 2:
    f_test = x3D * 0
    f_test = x3D.clone()
    df_dtheta = x3D * 0
    df_dlambda = x3D.new_zeros(x3D.numel())

elif IC == 4:
    xc = torch.tensor([
        math.sin(theta0) * math.cos(lambda0),
        math.sin(theta0) * math.sin(lambda0),
        math.cos(theta0)
    ], device=device)
    tmpx3D = x3D.T
    tmpy3D = y3D.T
    tmpz3D = z3D.T
    x_nodes = torch.stack([tmpx3D.reshape(-1), tmpy3D.reshape(-1), tmpz3D.reshape(-1)], dim=1)
    dotp = (x_nodes @ xc).clamp(-1.0, 1.0)
    alpha = torch.acos(dotp)

    #f_test = torch.zeros_like(x3D.T)
    f_test = torch.exp(-0.5 * (alpha / sigma)**2)
    f_test = f_test.reshape(x3D.T.shape)
    f_test = f_test.T

    lambda_nodes2D = torch.atan2(tmpy3D.reshape(-1), tmpx3D.reshape(-1))
    theta_nodes2D  = torch.acos(tmpz3D.reshape(-1))
    print("ETOV=", EToV[:10,:])
    print("VX = ", VX[:10])
    print("VY = ", VY[:10])
    print("VZ = ", VZ[:10])
    print("x3D = ", x3D[:,0])
    print("y3D = ", y3D[:,0])
    print("z3D = ", z3D[:,0])
    print("x3D flatten = ", tmpx3D.reshape(-1)[:10])
    print("y3D flatten = ", tmpy3D.reshape(-1)[:10])
    print("z3D flatten = ", tmpz3D.reshape(-1)[:10])
    print("lambda_nodes2D = ", lambda_nodes2D[:10])
    print("theta_nodes2D = ", theta_nodes2D[:10])
    sin_theta2D  = torch.sin(theta_nodes2D)
    cos_theta2D  = torch.cos(theta_nodes2D)
    sin_lambda2D = torch.sin(lambda_nodes2D)
    cos_lambda2D = torch.cos(lambda_nodes2D)
    print("sin_theta2D = ", sin_theta2D.shape)
    C = sin_theta2D * math.sin(theta0) * torch.cos(lambda_nodes2D - lambda0) + cos_theta2D * math.cos(theta0)
    C = C.clamp(-1.0, 1.0)
    alpha     = torch.acos(C)
    sqrt1mC2  = torch.sqrt(torch.clamp(1 - C**2, min=1e-10))
    prefactor = (alpha / sigma**2) * torch.exp(-0.5 * (alpha / sigma)**2) / sqrt1mC2

    df_dtheta = torch.zeros_like(x3D)
    print("df_theta = ", df_dtheta.shape)

    df_dtheta.view(-1)[:] = prefactor * (
        cos_theta2D * math.sin(theta0) * torch.cos(lambda_nodes2D - lambda0) -
        sin_theta2D * math.cos(theta0))
    print("df_theta = ", df_dtheta.shape)

    df_dlambda = torch.zeros_like(x3D.T, device=device)
    tmp = -prefactor * (sin_theta2D * math.sin(theta0) * torch.sin(lambda_nodes2D - lambda0))
    # reshape tmp to match df_dlambda without knowing the shape of df_dlambda
    tmp = tmp.reshape(x3D.T.shape)
    df_dlambda = tmp.T
    #print("tmp = ", tmp.shape)
    #print("tmp = ", tmp[:,:7])
    #df_dlambda.reshape(-1)[:] = -prefactor * (sin_theta2D * math.sin(theta0) * torch.sin(lambda_nodes2D - lambda0))
    #df_dlambda = df_dlambda.T

    # for each factor in above expression, print the first 10 values
    print("prefactor = ", prefactor.shape)
    print("prefactor = ", prefactor[:10])
    print("cos_theta2D = ", cos_theta2D[:10])
    print("sin_theta2D = ", sin_theta2D[:10])
    print("lambda_nodes2D = ", lambda_nodes2D[:10])
    print("lambda0 = ", lambda0)
    print("theta0 = ", theta0)
    print("df_dlambda = ", df_dlambda.shape)
    print("df_dlambda = ", df_dlambda[:,:10].cpu().numpy())
    print("df_dlambda, min = ", torch.min(df_dlambda))
    print("df_dlambda, max = ", torch.max(df_dlambda))
    print()

    # Check derivatives with finite differences

    epsilon = 1e-6
    lambda_plus  = lambda_nodes2D + epsilon
    lambda_minus = lambda_nodes2D - epsilon

    C_plus  = sin_theta2D * math.sin(theta0) * torch.cos(lambda_plus  - lambda0) + cos_theta2D * math.cos(theta0)
    C_minus = sin_theta2D * math.sin(theta0) * torch.cos(lambda_minus - lambda0) + cos_theta2D * math.cos(theta0)
    C_plus  = C_plus.clamp(-1.0, 1.0)
    C_minus = C_minus.clamp(-1.0, 1.0)
    print("C_plus = ", C_plus[:10])
    print("C_minus = ", C_minus[:10])

    if 0:
        # Generate some test numbers
        x = torch.linspace(-10, 10, steps=1000)  # values from -10 to 10

        # Save to file for MATLAB
        np.savetxt("x_values.txt", x.numpy())

        y_torch = torch.exp(x)

        # Load MATLAB results
        y_matlab = torch.tensor(np.loadtxt("matlab_exp.txt"))

        # Check difference
        max_abs_diff = torch.max(torch.abs(y_torch - y_matlab))
        print("Max absolute difference:", max_abs_diff.item())

        # Optional: Check relative error
        rel_error = torch.max(torch.abs((y_torch - y_matlab) / y_matlab))
        print("Max relative error:", rel_error.item())
        assert(1==0)

    f_plus  = torch.exp(-0.5 * (torch.acos(C_plus)  / sigma)**2)
    f_minus = torch.exp(-0.5 * (torch.acos(C_minus) / sigma)**2)
    print("f_plus = ", f_plus[:10])
    print("f_minus = ", f_minus[:10])   

    #df_dlambda_fd = torch.zeros_like(x3D.T)
    #df_dlambda_fd.reshape(-1)[:] = (f_plus - f_minus) / (2 * epsilon)
    #df_dlambda_fd = df_dlambda_fd.T
    #df_dlambda_fd = torch.zeros_like(x3D.T)
    df_dlambda_fd = (f_plus - f_minus) / (2 * epsilon)
    df_dlambda_fd = df_dlambda_fd.reshape(x3D.T.shape)
    df_dlambda_fd = df_dlambda_fd.T

    tmpdf_dlambda_fd = df_dlambda_fd.T
    tmpdf_dlambda    = df_dlambda.T
    diff_lambda = torch.max(torch.abs(tmpdf_dlambda.reshape(-1) - tmpdf_dlambda_fd.reshape(-1)))
    print(f'Max diff in df_dlambda: {diff_lambda:.4e}')
    print("df_dlambda = ", df_dlambda.shape)
    print("df_dlambda    = ", df_dlambda[:,:10].cpu().numpy())
    print("df_dlambda_fd = ", df_dlambda_fd[:,:10].cpu().numpy())

    print("df_lambda(51)    = %.6f"%(df_dlambda.T.reshape(-1)[50]))
    print("df_lambda_fd(51) = %.6f"%(df_dlambda_fd.T.reshape(-1)[50]))
    print("df_lambda(52)    = %.6f"%(df_dlambda.T.reshape(-1)[51]))
    print("df_lambda_fd(52) = %.6f"%(df_dlambda_fd.T.reshape(-1)[51]))
    for i in range(1000):
        # if error is larger than 0.01 then print the values
        if abs(df_dlambda.T.reshape(-1)[i] - df_dlambda_fd.T.reshape(-1)[i]) > 0.1:
            print(f"Node {i}: df_dlambda analytic = {df_dlambda.T.reshape(-1)[i]:.5f}, FD = {df_dlambda_fd.T.reshape(-1)[i]:.5f}")

    #assert(1==0)

    theta_plus  = theta_nodes2D + epsilon
    theta_minus = theta_nodes2D - epsilon

    C_plus  = torch.sin(theta_plus)  * math.sin(theta0) * torch.cos(lambda_nodes2D - lambda0) + torch.cos(theta_plus)  * math.cos(theta0)
    C_minus = torch.sin(theta_minus) * math.sin(theta0) * torch.cos(lambda_nodes2D - lambda0) + torch.cos(theta_minus) * math.cos(theta0)
    C_plus  = C_plus.clamp(-1.0, 1.0)
    C_minus = C_minus.clamp(-1.0, 1.0)

    alpha_plus  = torch.acos(C_plus)
    alpha_minus = torch.acos(C_minus)
    f_plus  = torch.exp(-0.5 * (alpha_plus  / sigma)**2)
    f_minus = torch.exp(-0.5 * (alpha_minus / sigma)**2)

    df_dtheta_fd = torch.zeros_like(x3D.T)
    df_dtheta_fd.reshape(-1)[:] = (f_plus - f_minus) / (2 * epsilon)
    df_dtheta_fd = df_dtheta_fd.T

    tmpdf_dtheta_fd = df_dtheta_fd.T
    tmpdf_dtheta = df_dtheta.T
    error_theta = torch.max(torch.abs(tmpdf_dtheta.reshape(-1) - tmpdf_dtheta_fd.reshape(-1)))
    print(f'Max diff in df_dtheta (FD check): {error_theta:.4e}')

    #assert(1==0)

# === Time Integration and Advection Solver in PyTorch ===

# === Advection RHS Function ===
#def Advec2d_sphereRHS_CARTESIAN_apek(u, v_x, v_y, v_z, sn_x, sn_y, sn_z):
#    global nx, ny, nz, rx2D, ry2D, sx2D, sy2D, rz2D, sz2D, J2D
#    global MassMatrix2D, Dr2D, Ds2D, Ic2f2D, MassMatrix2Dfine, IIf
#    global Dr2Df, Ds2Df, x3D, y3D, z3D, vmapM, vmapP, LIFT, Fscale
def Advec2d_sphereRHS_CARTESIAN_vectorized(
    u, v_x, v_y, v_z, sn_x, sn_y, sn_z, nx, ny, nz,
    rx2D, ry2D, sx2D, sy2D, rz2D, sz2D, J2D,      # J2D not needed algebraically after cancellation, but kept for API parity
    MassMatrix2D, Dr2D, Ds2D,
    LIFT, Fscale, vmapM, vmapP):
    """
    Vectorized RHS for advection on the sphere.
    No per-element loops, no diag(), no solves.
    """

    # --- 1) Project velocity onto the tangent plane (elementwise) ---
    c_dot_sn = v_x*sn_x + v_y*sn_y + v_z*sn_z
    vxt = v_x - c_dot_sn*sn_x
    vyt = v_y - c_dot_sn*sn_y
    vzt = v_z - c_dot_sn*sn_z

    # --- 2) Face fluxes (one big gather, no loops) ---
    # flatten in element-major order: (Np,K) -> (K*Np,)
    # we use .transpose(0,1).reshape(-1) to match your previous indexing
    vxt_flat = vxt.transpose(0,1).reshape(-1)
    vyt_flat = vyt.transpose(0,1).reshape(-1)
    vzt_flat = vzt.transpose(0,1).reshape(-1)
    u_flat   = u  .transpose(0,1).reshape(-1)

    vxtM = vxt_flat[vmapM]; vytM = vyt_flat[vmapM]; vztM = vzt_flat[vmapM]
    #vxtP = vxt_flat[vmapP]; vytP = vyt_flat[vmapP]; vztP = vzt_flat[vmapP]

    nx_flat = nx.transpose(0,1).reshape(-1)
    ny_flat = ny.transpose(0,1).reshape(-1)
    nz_flat = nz.transpose(0,1).reshape(-1)

    vM_n = vxtM*nx_flat + vytM*ny_flat + vztM*nz_flat

    uM = u_flat[vmapM]
    uP = u_flat[vmapP]

    # Central flux (switch to upwind if you prefer)
    f     = vM_n * uM
    #fstar = vM_n * (uM + uP) * 0.5
    mask = vM_n >= 0
    fstar = torch.where(mask, 
                        vM_n * uM,  # Upwind: use upstream value
                        vM_n * uP)  # Upwind: use upstream value    

    df = (f - fstar).view(nx.shape[1], nx.shape[0]).t()   # (3*Nfp, K)

    # --- 3) Volume contribution (all vectorized) ---
    # Gradients in reference directions applied to u: (Np,Np)@(Np,K) -> (Np,K)
    du_r = Dr2D @ u
    du_s = Ds2D @ u

    # Map to Cartesian components using metric arrays (elementwise)
    gx = rx2D*du_r + sx2D*du_s     # (Np,K)
    gy = ry2D*du_r + sy2D*du_s
    gz = rz2D*du_r + sz2D*du_s

    # Project gradient onto tangent plane: g_t = g - (sn·g) sn
    n_dot_g = sn_x*gx + sn_y*gy + sn_z*gz
    px = gx - sn_x*n_dot_g
    py = gy - sn_y*n_dot_g
    pz = gz - sn_z*n_dot_g

    # Apply mass matrix on each component (left multiply), then weight by tangential velocity
    # NOTE: In your loop you formed M_k = diag(J) @ M and later solved M_k \ (...).
    #       Those J's cancel analytically: rhs_vol = invM * ( vxt*(M@px) + ... ).

    #Mpx = MassMatrix2D @ px        # (Np,K)
    #Mpy = MassMatrix2D @ py
    #Mpz = MassMatrix2D @ pz
    #vol = vxt*Mpx + vyt*Mpy + vzt*Mpz  # (Np,K)
    #rhs_vol = invM2D @ vol             # (Np,K)
    vol = vxt*px + vyt*py + vzt*pz  # (Np,K)
    rhs_vol = vol             # (Np,K)

    # --- 4) Face lifting (already batched) ---
    rhs_face = LIFT @ (Fscale * df)    # (Np,K)

    # --- 5) Final RHS ---
    rhsu = -rhs_vol + rhs_face
    return rhsu

torch._dynamo.config.suppress_errors = True  # don’t crash if a path is unsupported
compiled_rhs = torch.compile(Advec2d_sphereRHS_CARTESIAN_vectorized, dynamic=True)

def Advec2d_sphereRHS_CARTESIAN_apek_WORKING(
    u, v_x, v_y, v_z, sn_x, sn_y, sn_z, nx, ny, nz,
    rx2D, ry2D, sx2D, sy2D, rz2D, sz2D, J2D,
    MassMatrix2D, Dr2D, Ds2D,
    LIFT, Fscale, vmapM, vmapP):

    # Project velocity onto tangent space
    c_dot_sn = v_x * sn_x + v_y * sn_y + v_z * sn_z
    vxt = v_x - c_dot_sn * sn_x
    vyt = v_y - c_dot_sn * sn_y
    vzt = v_z - c_dot_sn * sn_z

    vxt_flat = vxt.T.reshape(-1)
    vyt_flat = vyt.T.reshape(-1)
    vzt_flat = vzt.T.reshape(-1)
    vxtM = vxt_flat[vmapM]
    vytM = vyt_flat[vmapM]
    vztM = vzt_flat[vmapM]
    vxtP = vxt_flat[vmapP]
    vytP = vyt_flat[vmapP]
    vztP = vzt_flat[vmapP]

    nx_flat = nx.T.reshape(-1)
    ny_flat = ny.T.reshape(-1)
    nz_flat = nz.T.reshape(-1)
    nxM = nx_flat[:]
    nyM = ny_flat[:]
    nzM = nz_flat[:]

    vM_n = vxtM * nxM + vytM * nyM + vztM * nzM
    #vP_n = vxtP * nxM + vytP * nyM + vztP * nzM

    #uM = u.flatten(start_dim=0)[vmapM]
    #uP = u.flatten(start_dim=0)[vmapP]
    uM = u.T.reshape(-1)[vmapM]
    uP = u.T.reshape(-1)[vmapP]
    f = vM_n * uM
    fstar = vM_n * (uM + uP) * 0.5

    # Replace your flux calculation
    #fstar = torch.where(vM_n >= 0, 
    #                    vM_n * uM,  # Upwind: use upstream value
    #                    vM_n * uP)  # Upwind: use upstream value    
    df = f - fstar
    df = df.reshape(nx.T.shape)
    df = df.T  # Transpose to match original shape

    rhsu = torch.zeros_like(u, device=device)
    Ddir2D = torch.zeros_like(u, device=device)

    # print("snx=", sn_x.shape)
    # print("v_x=", v_x.shape)
    # print("nx=", nx.shape)
    # print("vmapM=", vmapM.shape)
    # print("vxtM=", vxtM.shape)
    # print("nxM=", nxM.shape)
    # print("vM_n=", vM_n.shape)
    # print("uM=", uM.shape)
    # print("df.shape=",df.shape)
    # print("rhsu.shape=",rhsu.shape)
    # print("LIFT.shape=",LIFT.shape)
    #print("Fscale.shape=",Fscale.shape)
    #print("Fscale = ", Fscale[:,:5])

    #print("sJ.shape=",sJ.shape)
    #print("sJ = ", sJ[:,:5])
    #print("u.shape=", u.shape)
    #print("u= ", u[:,:7].cpu().numpy())
    #print("u flat=", u.flatten(start_dim=0)[:5].cpu().numpy())
    #print("uM.shape=", uM.shape)
    #print("uM = ", uM[:70].cpu().numpy())
    #print("vmapM=", vmapM[:70])
    #assert(1==0)
    for k in range(K):
        # diag(J2D(:,k)) * MassMatrix2D;

        Jk = J2D[:,k].diag()
        M2Dk = Jk @ MassMatrix2D

        snx  = sn_x[:, k].diag()
        sny  = sn_y[:, k].diag()
        snz  = sn_z[:, k].diag()
        vxtk = vxt[:, k].diag()
        vytk = vyt[:, k].diag()
        vztk = vzt[:, k].diag()

        if 0:
            print("rx2D.shape=", rx2D.shape)
            print("ry2D.shape=", ry2D.shape)
            print("sx2D.shape=", sx2D.shape)
            print("sy2D.shape=", sy2D.shape)
            print("rz2D.shape=", rz2D.shape)
            print("sz2D.shape=", sz2D.shape)
            print("rx2D[:,k]=", rx2D[:, k])
            print("ry2D[:,k]=", ry2D[:, k])
            print("sx2D[:,k]=", sx2D[:, k])
            print("sy2D[:,k]=", sy2D[:, k])
            print("rz2D[:,k]=", rz2D[:, k])
            print("sz2D[:,k]=", sz2D[:, k])
            # return the diagonal elements
            print("snx=", snx.diagonal())
            print("sny=", sny.diagonal())
            print("snz=", snz.diagonal())
            print("vxtk=", vxtk.diagonal())
            print("vytk=", vytk.diagonal())
            print("vztk=", vztk.diagonal())

            #assert(1==0)
            
        Dxk = torch.diag(rx2D[:, k]) @ Dr2D + torch.diag(sx2D[:, k]) @ Ds2D
        Dyk = torch.diag(ry2D[:, k]) @ Dr2D + torch.diag(sy2D[:, k]) @ Ds2D
        Dzk = torch.diag(rz2D[:, k]) @ Dr2D + torch.diag(sz2D[:, k]) @ Ds2D

        if 0:
            grad_ux = Dxk @ u[:, k]
            grad_uy = Dyk @ u[:, k]
            grad_uz = Dzk @ u[:, k]

            n_dot_grad_u = snx * grad_ux + sny * grad_uy + snz * grad_uz
            proj_grad_x  = grad_ux - snx * n_dot_grad_u
            proj_grad_y  = grad_uy - sny * n_dot_grad_u
            proj_grad_z  = grad_uz - snz * n_dot_grad_u

            advec = vxtk * proj_grad_x + vytk * proj_grad_y + vztk * proj_grad_z
            # Ddir2D(:,k) = M2Dk \ advec
            Ddir2D[:, k] = advec # torch.linalg.solve(M2Dk, advec)
        else:
            n_dot_grad  = snx @ Dxk + sny @ Dyk + snz @ Dzk
            proj_grad_x = Dxk - snx @ n_dot_grad
            proj_grad_y = Dyk - sny @ n_dot_grad
            proj_grad_z = Dzk - snz @ n_dot_grad
            
            #advecx = Jk @ vxtk @ MassMatrix2D @ proj_grad_x 
            #advecy = Jk @ vytk @ MassMatrix2D @ proj_grad_y 
            #advecz = Jk @ vztk @ MassMatrix2D @ proj_grad_z
            advecx = vxtk @ M2Dk @ proj_grad_x 
            advecy = vytk @ M2Dk @ proj_grad_y 
            advecz = vztk @ M2Dk @ proj_grad_z

            uk = u[:, k]
            advec = (advecx + advecy + advecz ) @ uk
            # Ddir2D[:,k] = torch.linalg.solve(M2Dk, advec) 
            Ddir2D[:,k] = invM2D @ (advec / J2D[:,k])

    rhsu = -Ddir2D + LIFT @ (Fscale * df)

    if 0:
        print("rhsu.shape=", rhsu.shape)
        print("rhsu = ", rhsu[:,:10].cpu().numpy())
        tmp = LIFT @ (Fscale * df)
        print("Ddir2D = ", Ddir2D[:,:10].cpu().numpy())
        print("tmp.shape=", tmp.shape)
        print("tmp = ", tmp[:,:10].cpu().numpy())
        print("max Ddir2D = ", torch.max(Ddir2D), "min Ddir2D = ", torch.min(Ddir2D))
        plt.pause(0.01)
        #assert(1==0)
    return rhsu

# === Advection field ===
omega = 0.4
v_x = -omega * y3D
v_y = omega * x3D
v_z = torch.zeros_like(x3D)

# === Time parameters ===
time = 0.0
FinalTime = 2 * math.pi / v_phi
Nsteps = round(FinalTime / 0.01)
CFL = 2*0.25
h = 2*np.pi / np.sqrt(K)  # Approximate grid spacing on the sphere
vel_mag = torch.sqrt(v_x**2 + v_y**2 + v_z**2)
vmax = torch.max(vel_mag)
dt = CFL * h / ((2*N+1)*vmax)
Nsteps = torch.round( FinalTime / dt)
dt = FinalTime / Nsteps
print("dt = ", dt)

# === Compute unit normals to the sphere ===
norm_sn = torch.sqrt(x3D**2 + y3D**2 + z3D**2)
sn_x = x3D / norm_sn
sn_y = y3D / norm_sn
sn_z = z3D / norm_sn

# === Setup initial condition ===
u = torch.zeros_like(x3D,device=device)
u.copy_(f_test)
#u = x3D.clone() + y3D.clone()  # Simple linear function in x-direction

#u = x3D.clone()
#v_x = torch.ones_like(x3D)
#v_y = torch.zeros_like(x3D)
#v_z = torch.zeros_like(x3D)

# Flatten the arrays (like x3D(:) in MATLAB)
x_flat = x3D.reshape(-1).cpu().numpy()
y_flat = y3D.reshape(-1).cpu().numpy()
z_flat = z3D.reshape(-1).cpu().numpy()

# store u in a .mat file to be loaded in matlab workspace
#    import scipy.io as sio
if 0:
    sio.savemat('u_initial.mat', {'u': u.cpu().numpy(), 'x3D': x3D.cpu().numpy(), 'y3D': y3D.cpu().numpy(), 'z3D': z3D.cpu().numpy()})
    print('Initial condition saved to u_initial.mat')
# === Filter setup ===  
# assert(1==0)
# === Runge-Kutta coefficients ===
rk4a = [0.0, -567301805773.0/1357537059087.0, -2404267990393.0/2016746695238.0,
         -3550918686646.0/2091501179385.0, -1275806237668.0/842570457699.0]
rk4b = [1432997174477.0/9575080441755.0, 5161836677717.0/13612068292357.0,
         1720146321549.0/2090206949498.0, 3134564353537.0/4481467310338.0,
         2277821191437.0/14882151754819.0]

# === RK storage ===
resu = torch.zeros_like(u, device=device)

if MIXEDONOFF:
    # constants
    Dr2D         = Dr2D.to(amp_dtype)
    Ds2D         = Ds2D.to(amp_dtype)
    MassMatrix2D = MassMatrix2D.to(amp_dtype)
    invM2D       = invM2D.to(amp_dtype)  # or: invM2D.to(amp_dtype) if you already have invM2D

    LIFT         = LIFT.to(amp_dtype)
    Fscale       = Fscale.to(amp_dtype)

    # metrics & normals
    rx2D, ry2D, rz2D = rx2D.to(amp_dtype), ry2D.to(amp_dtype), rz2D.to(amp_dtype)
    sx2D, sy2D, sz2D = sx2D.to(amp_dtype), sy2D.to(amp_dtype), sz2D.to(amp_dtype)
    J2D = J2D.to(amp_dtype)  # kept for API parity even if it cancels out
    nx, ny, nz       = nx.to(amp_dtype), ny.to(amp_dtype), nz.to(amp_dtype)
    sn_x, sn_y, sn_z = sn_x.to(amp_dtype), sn_y.to(amp_dtype), sn_z.to(amp_dtype)

    # state & velocity fields (cast once, keep them in AMP dtype during the loop)
    u      = u.to(amp_dtype)
    v_x    = v_x.to(amp_dtype)
    v_y    = v_y.to(amp_dtype)
    v_z    = v_z.to(amp_dtype)
    resu   = resu.to(amp_dtype)   # RK accumulator
    dt = dt.to(amp_dtype)  # time step

# === Time stepping loop ===
tstep = 0
# Plotting
#fig = plt.figure(figsize=(8, 6))
#ax = fig.add_subplot(111, projection='3d')
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# warmup before timing
for _ in range(5):
    #rhsu = Advec2d_sphereRHS_CARTESIAN_apek(u, v_x, v_y, v_z, sn_x, sn_y, sn_z)
    with autocast_ctx:
        rhsu = compiled_rhs( # Advec2d_sphereRHS_CARTESIAN_vectorized(
            u, v_x, v_y, v_z, sn_x, sn_y, sn_z, nx, ny, nz,
            rx2D, ry2D, sx2D, sy2D, rz2D, sz2D, J2D,
            MassMatrix2D, Dr2D, Ds2D,
            LIFT, Fscale, vmapM, vmapP)

# start timer
torch.mps.synchronize()  # ensure GPU is idle before starting
t0 = timing.perf_counter()

print(f'Total No. Elements: {K}, Total DOFs: {K*Np}')
while time < FinalTime:
    tstep += 1
    print(f'Time step {tstep}/{Nsteps}, time = {time:.3f}, dt = {dt:.3f}')
    # output min and max of u
    print(f'  Min u: {u.min().item():.3e}, Max u: {u.max().item():.3e}')
    #print(f'  u:',u[:,:10].cpu().numpy())
    if tstep % PLOTFREQ == 0:

        #ax.cla()  # clear the previous plot

        u_flat = u.detach().reshape(-1).cpu().numpy()

        # 3D scatter plot
        sc = ax.scatter(x_flat, y_flat, z_flat, c=u_flat, s=50, cmap='jet',vmin=0,vmax=1)

        # Appearance settings
        fig.patch.set_facecolor('white')
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])  # axis equal

        # Color scale
        #sc.set_clim(0, 1)  # fix color axis range
        # plt.colorbar(sc)  # optional, like `colorbar` in MATLAB

        plt.title(f"Solution at t = {time:.4f}, step: {tstep}")
        plt.draw()

        # -- inside the loop --
        #u_flat = u.detach().reshape(-1).cpu().numpy()
        #sc.set_array(u_flat) # update colors
        # if x/y/z move too:
        # sc._offsets3d = (x_flat, y_flat, z_flat)
        #fig.canvas.draw_idle()
        plt.pause(0.001)  # Like `drawnow` in MATLAB

    # Adjust dt if it exceeds the final time
    if tstep == Nsteps:
        dt = FinalTime - time

    for INTRK in range(5):
        #rhsu = Advec2d_sphereRHS_CARTESIAN_apek(u, v_x, v_y, v_z, sn_x, sn_y, sn_z)
        with autocast_ctx:
            #rhsu = Advec2d_sphereRHS_CARTESIAN_vectorized(
            rhsu = compiled_rhs( # Advec2d_sphereRHS_CARTESIAN_vectorized(
                u, v_x, v_y, v_z, sn_x, sn_y, sn_z, nx, ny, nz,
                rx2D, ry2D, sx2D, sy2D, rz2D, sz2D, J2D,
                MassMatrix2D, Dr2D, Ds2D,
                LIFT, Fscale, vmapM, vmapP)

        #rhsu = Advec2d_sphereRHS_CARTESIAN_apek(
        #    u, v_x, v_y, v_z, sn_x, sn_y, sn_z, nx, ny, nz,
        #    rx2D, ry2D, sx2D, sy2D, rz2D, sz2D, J2D,
        #    MassMatrix2D, Dr2D, Ds2D,
        #    LIFT, Fscale, vmapM, vmapP)
        #rhsu += 0.01 * torch.randn_like(u)  # add noise
        resu = rk4a[INTRK] * resu + dt * rhsu
        #u_prev = u.clone()
        u = u + rk4b[INTRK] * resu
        if 0:
            print("rk4a=", rk4a[INTRK])
            print("rk4b=", rk4b[INTRK])
            #print("u=", u[:,:10].cpu().numpy())
            print("rhsu=", resu[:,:10].cpu().numpy())
            print("resu=", resu[:,:10].cpu().numpy())
            plt.pause(0.01)
            delta_u = torch.norm(u - u_prev) / torch.norm(u)
            print(f"[RKStep {INTRK}] Δu_rel = {delta_u:.2e}")
            u_prev = u.clone()
            print("dt=",dt)
            print("Velocity max:", v_x.abs().max().item(), v_y.abs().max().item(), v_z.abs().max().item())
            print("sn_x max:", sn_x.abs().max().item())
            print("sn_y max:", sn_y.abs().max().item())
            print("sn_z max:", sn_z.abs().max().item())
            print("rhsu min:", rhsu.abs().min().item())
            print("rhsu max:", rhsu.abs().max().item())

    if globals.filterONOFF:
        print('Modal filter applied...')
        u = Filter2D @ u

    time += dt

torch.mps.synchronize()  # wait for GPU to finish
t1 = timing.perf_counter()

print(f'Total No. Elements: {K}, Total DOFs: {K*Np}')
print(f"Simulation took {t1 - t0:.3f} seconds")