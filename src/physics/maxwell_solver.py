# Solver de Maxwell DIFERENCIÁVEL (substitui o adjunto de difusão quebrado).
# Construído sobre o FDFDSolver de Maxwell já existente (src/physics/fdfd.py),
# com gradiente adjunto exato do sistema linear complexo A·Ez = b.

import torch
import numpy as np
import scipy.sparse.linalg as spla
from src.physics.fdfd import FDFDSolver


class MaxwellTransmission(torch.autograd.Function):
    """
    forward:  eps (B, ny, nx) real + FDFDSolver -> transmissão (B, 2) = [Re, Im].
              ESSA é a predição do PEDS (saída do solver de Maxwell).
    backward: estado adjunto. Para t = mᵀ·A⁻¹·b com A(ε)=L-ω²·diag(ε):
                  ∂t/∂ε_k = ω²·Ez_k·(Aᵀ⁻¹·m)_k
              Resolve UM sistema extra Aᵀ·λ = m por amostra (sem inverter A).
    """

    @staticmethod
    def forward(ctx, eps_batch: torch.Tensor, solver: FDFDSolver) -> torch.Tensor:
        eps_np = eps_batch.detach().cpu().numpy().astype(np.float64)
        if eps_np.ndim == 2:
            eps_np = eps_np[None, ...]
        B = eps_np.shape[0]
        ny, nx = solver.sd.ny, solver.sd.nx

        b = solver.get_continuous_source()
        mask = solver.get_monitor_mask().flatten()
        nmon = int(mask.sum())
        m_vec = np.zeros(ny * nx, dtype=np.complex128)
        m_vec[mask] = 1.0 / nmon
        omega2 = float(solver.sd.omega) ** 2

        out = np.zeros((B, 2), dtype=np.float64)
        A_list, Ez_list = [], []
        for i in range(B):
            A = solver.build_maxwell_operator(eps_np[i].reshape(ny, nx))
            Ez = spla.spsolve(A, b)
            t = complex(m_vec @ Ez)
            out[i, 0] = t.real
            out[i, 1] = t.imag
            A_list.append(A)
            Ez_list.append(Ez)

        ctx.A_list = A_list
        ctx.Ez_list = Ez_list
        ctx.m_vec = m_vec
        ctx.omega2 = omega2
        ctx.shape = (ny, nx)
        ctx.device = eps_batch.device
        ctx.dtype = eps_batch.dtype
        ctx.was_2d = (eps_batch.dim() == 2)
        return torch.tensor(out, dtype=eps_batch.dtype, device=eps_batch.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        g = grad_output.detach().cpu().numpy().astype(np.float64)
        if g.ndim == 1:
            g = g[None, :]
        ny, nx = ctx.shape
        n = ny * nx
        B = len(ctx.A_list)

        grad_eps = np.zeros((B, n), dtype=np.float64)
        for i in range(B):
            A = ctx.A_list[i]
            Ez = ctx.Ez_list[i]
            lam = spla.spsolve(A.T, ctx.m_vec)      # Aᵀ·λ = m
            dt = ctx.omega2 * (Ez * lam)            # ∂t/∂ε_k (complexo)
            grad_eps[i] = g[i, 0] * dt.real + g[i, 1] * dt.imag

        grad = torch.tensor(
            grad_eps.reshape(B, ny, nx), dtype=ctx.dtype, device=ctx.device
        )
        if ctx.was_2d:
            grad = grad.squeeze(0)
        return grad, None


class DifferentiableMaxwell(torch.nn.Module):
    """Recebe geometria combinada (B, ny, nx) e devolve (rp, ip) diferenciáveis."""
    def __init__(self, solver: FDFDSolver):
        super().__init__()
        self.solver = solver

    def forward(self, eps_batch: torch.Tensor):
        out = MaxwellTransmission.apply(eps_batch, self.solver)
        return out[:, 0], out[:, 1]
