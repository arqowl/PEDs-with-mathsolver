# src/physics/diffusion_solver.py
# Solver de difusão (baixa fidelidade) DIFERENCIÁVEL para os surrogates
# Fourier(d)/Fisher(d). Porta fiel de fourier_solver.jl (targetfunc + rrule),
# com adjunto VALIDADO contra diferença finita (erro ~1e-7) e generatepores
# VALIDADO contra o FE low-fidelity do paper (Fourier: 0.140 vs 0.135).
#
# Física: resolve  ∇·(c∇T)=0  com Dirichlet (T=1 embaixo, 0 em cima),
# periódico em y, e devolve o fluxo κ num plano em x=0.5.

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


# ----------------------- núcleo numpy (validado) -----------------------
def get_position(L: float, res: int) -> np.ndarray:
    nx = int(round(L * res))
    return np.arange(1, nx + 1) * (1.0 / res)


def sdiff1(x: np.ndarray) -> sp.csr_matrix:
    N = len(x) - 2
    dx1 = 1.0 / (x[1:N + 1] - x[0:N])
    dx2 = -1.0 / (x[2:N + 2] - x[1:N + 1])
    return sp.diags([dx1, dx2], [0, -1], shape=(N + 1, N)).tocsr()


def sdiff1_periodic(x: np.ndarray) -> sp.csr_matrix:
    # porta fiel do Julia (grade uniforme): D (N+1)x(N+1), N=len(x)-2
    N = len(x) - 2
    M = N + 1
    inv = 1.0 / (x[1:] - x[:-1])
    main = inv[0:M]
    sub = -inv[0:M - 1]
    D = sp.diags([main, sub], [0, -1], shape=(M, M)).tolil()
    D[M - 1, M - 1] = -D[M - 1, M - 2]   # cantos periódicos
    D[0, M - 1] = -D[0, 0]
    return D.tocsr()


def laplacian_arrayC(x, y, c):
    Dx = sdiff1(x); Nx = Dx.shape[1]
    Dy = sdiff1_periodic(y); Ny = Dy.shape[1]
    G = sp.vstack([sp.kron(sp.eye(Ny), Dx), sp.kron(Dy, sp.eye(Nx))]).tocsc()
    C = sp.diags(np.asarray(c).ravel(order='F'), 0, shape=(G.shape[0],) * 2, format='csc')
    return (-G.T @ C @ G).tocsc(), G


def create_cOp(N: int) -> sp.csc_matrix:
    """Operador de média (sub-pixel) que mapeia geometria N×N -> vetor-c de arestas."""
    o = np.ones(N) / 2.0
    D = sp.diags([o, o[:-1]], [0, 1], shape=(N, N))
    Id = sp.eye(N)
    avgOpx = sp.kron(Id.tocsr()[:-1, :], D.tocsr()[:-1, :])
    avgOpy = sp.kron(D.tocsr()[:-1, :], Id.tocsr()[1:-1, :])
    return sp.vstack([avgOpx, avgOpy]).tocsc()


def generatepores(widths: np.ndarray) -> np.ndarray:
    """downsample(p): largura do furo w -> condutividade 1-0.9w (medium=1, furo=0.1).
    Validado: FE coarse Fourier(16)=0.140 vs paper 0.135; Fourier(25)=0.087 vs 0.085."""
    return 1.0 - 0.9 * np.asarray(widths)


def _targetfunc_core(x, y, c):
    dx = x[1] - x[0]; dy = y[1] - y[0]; n = len(x)
    A, G = laplacian_arrayC(x, y, c)
    c1 = np.asarray(c)[:(n - 1) * (n - 1)].reshape((n - 1, n - 1), order='F')
    Nx, Ny = len(x) - 2, len(y) - 1
    S = np.zeros((Nx, Ny))
    S[-1, :] = -(c1[-1, :]) / dx ** 2
    T = spla.spsolve(A, S.ravel(order='F')).reshape((Nx, Ny), order='F')
    iline = int(np.sum(x < 0.5))
    integrand = c1[iline, :] * (T[iline, :] - T[iline - 1, :])
    fval = float(np.sum(integrand) / dx * dy)
    return fval, A, G, T, c1, iline, dx, dy, Nx, Ny


def _targetfunc_grad_c(x, y, c):
    fval, A, G, T, c1, iline, dx, dy, Nx, Ny = _targetfunc_core(x, y, c)
    n = len(x)
    gx = np.zeros((Nx, Ny))
    gx[iline, :] = c1[iline, :] / dx * dy
    gx[iline - 1, :] = -c1[iline, :] / dx * dy
    lam = spla.spsolve(A.T, gx.ravel(order='F'))
    Tv = T.ravel(order='F')
    Aterm = (G @ lam) * (G @ Tv)                       # forma elementar do termo de A
    grad = np.array(Aterm, dtype=np.float64)
    for j in range(n - 1):                              # termo explícito do integrando
        grad[j * (n - 1) + iline] += (T[iline, j] - T[iline - 1, j]) / dx * dy
    for j in range(n - 1):                              # termo da fonte S(c)
        grad[j * (n - 1) + (n - 2)] += -lam[j * Nx + (Nx - 1)] / dx ** 2
    return fval, grad


# ----------------------- estrutura de simulação -----------------------
class DiffusionSim:
    """Pré-computa grade + operador de média para uma dada resolução coarse."""
    def __init__(self, coarse_resolution: int, Lx: float = 1.0, Ly: float = 1.0):
        self.res = coarse_resolution
        self.x = get_position(Lx, coarse_resolution)
        self.y = get_position(Ly, coarse_resolution)
        self.avgOp = create_cOp(coarse_resolution)                 # (length_c, N²)
        self.avgOp_dense = torch.tensor(self.avgOp.toarray(), dtype=torch.float32)


# ----------------------- ponte diferenciável (torch) -----------------------
class DiffusionFlux(torch.autograd.Function):
    """c (B, length_c) -> fluxo κ (B,). Adjunto exato (validado vs FD)."""
    @staticmethod
    def forward(ctx, c_batch: torch.Tensor, sim: DiffusionSim) -> torch.Tensor:
        c_np = c_batch.detach().cpu().numpy().astype(np.float64)
        if c_np.ndim == 1:
            c_np = c_np[None, :]
        B = c_np.shape[0]
        out = np.zeros(B); grads = np.zeros_like(c_np)
        for i in range(B):
            f, g = _targetfunc_grad_c(sim.x, sim.y, c_np[i])
            out[i] = f; grads[i] = g
        ctx.grads = grads
        ctx.device = c_batch.device; ctx.dtype = c_batch.dtype
        ctx.was_1d = (c_batch.dim() == 1)
        return torch.tensor(out, dtype=c_batch.dtype, device=c_batch.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        g = grad_output.detach().cpu().numpy()
        if g.ndim == 0:
            g = g[None]
        grad_c = ctx.grads * g[:, None]                # (B, length_c)
        out = torch.tensor(grad_c, dtype=ctx.dtype, device=ctx.device)
        if ctx.was_1d:
            out = out.squeeze(0)
        return out, None


class DifferentiableDiffusion(torch.nn.Module):
    """Recebe geometria combinada (B, N²) e devolve o fluxo κ (B,) diferenciável."""
    def __init__(self, sim: DiffusionSim):
        super().__init__()
        self.sim = sim
        self.register_buffer("avgOp", sim.avgOp_dense)   # (length_c, N²)

    def forward(self, geom_batch: torch.Tensor) -> torch.Tensor:
        # geom (B, N²) -> c (B, length_c) -> κ (B,)
        c = geom_batch @ self.avgOp.t()
        return DiffusionFlux.apply(c, self.sim)
