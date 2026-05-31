import torch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import Dict
from .fourier_solver import laplacian

# Assumindo que a função laplacian e outras matrizes já estão definidas
# from .fourier_solver import laplacian

@dataclass
class Simulation:
    """
    Equivalente à struct 'Simulation' no Julia.
    Guarda o domínio espacial e as matrizes derivadas pré-computadas (Ap e bp)
    para acelerar o cálculo do gradiente adjunto no backward pass.
    """
    x: np.ndarray
    y: np.ndarray
    Ap: Dict[int, sp.csc_matrix]
    bp: Dict[int, np.ndarray]

class TargetFuncAdjoint(torch.autograd.Function):
    """
    Ponte Diferenciável entre o Simulador Físico (SciPy) e a Rede Neuronal (PyTorch).
    Implementa o Método do Estado Adjunto (Adjoint Method) para derivar Ax = b.
    """
    
    @staticmethod
    def forward(ctx, c_tensor: torch.Tensor, sim: Simulation) -> torch.Tensor:
        c = c_tensor.detach().cpu().numpy()
        x, y = sim.x, sim.y
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # 1. Construir Laplaciano
        A = laplacian(x, y, c, periodicy=True, arrayC=True)
        
        # 2. DIMENSÕES REAIS: Use as dimensões do grid original
        # O solver laplaciano no seu código usa len(x) e len(y)
        nx = len(x)
        ny = len(y)
        
        # 3. Preparar fonte (S) de forma segura
        # Se A tem num_vars linhas, S deve ter esse tamanho
        num_vars = A.shape[0]
        S = np.zeros(num_vars)
        # Aplica a fonte na borda baseada no tamanho do grid real
        offset = (nx - 1) 
        S[-offset:] = -c.flatten()[-offset:] / (dx**2)
        
        # 4. Resolver
        T_flat = spla.spsolve(A, S)
        
        # 5. Avaliar fval (Reshape usando as dimensões reais nx, ny)
        iline = ny // 2
        T_grid = T_flat.reshape((nx - 2, ny - 1)) # Ajuste conforme o sdiff1 que você usa
        c_grid = c.reshape((nx, ny))              # O c original deve casar com o grid
        
        # Ajuste o slice do integrand para o tamanho de T_grid
        integrand = c_grid[iline, :len(T_grid[iline, :])] * (T_grid[iline, :] - T_grid[iline - 1, :])
        fval = np.sum(integrand) / dx * dy
        
        # 6. Guardar estado para o backward pass
        ctx.sim = sim
        ctx.dx = dx
        ctx.dy = dy
        ctx.iline = iline
        ctx.A = A
        ctx.T_flat = T_flat
        ctx.c_grid = c_grid
        
        return torch.tensor(fval, dtype=torch.float32, device=c_tensor.device, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        O "Backward Pass" (Regra da Cadeia). Corresponde ao `ChainRules.rrule`.
        Calcula o gradiente adjunto usando a transposta do Laplaciano.
        """
        # 1. Recuperar o estado da simulação guardado no forward
        sim = ctx.sim
        dx, dy, iline = ctx.dx, ctx.dy, ctx.iline
        A, T_flat, c1 = ctx.A, ctx.T_flat, ctx.c1
        n = len(sim.x)
        
        # 2. Construir o vetor de gradiente do campo (gx)
        gx = np.zeros((len(sim.x) - 2, len(sim.y) - 1))
        gx[iline, :]     =  c1[iline, :] / dx * dy
        gx[iline - 1, :] = -c1[iline, :] / dx * dy
        
        # 3. O SEGREDO DO ADJUNTO: Resolver λ = A^T \ gx
        # Evitamos derivar a inversa da matriz; resolvemos apenas um novo sistema linear!
        lambda_vec = spla.spsolve(A.T, gx.flatten())
        
        # 4. Derivada explícita de fval em relação a c (fc)
        T_mat = T_flat.reshape((len(sim.x) - 2, len(sim.y) - 1))
        fc = np.zeros(len(ctx.c1.flatten()) + (n-2)*(n-1)) # O mesmo tamanho do vetor 'c' completo
        
        for j in range(n - 1):
            # Acesso sequencial garantindo que preenchemos na mesma ordem do flatten
            idx = j * (n - 1) + iline
            if idx < len(fc):
                fc[idx] = (T_mat[iline, j] - T_mat[iline - 1, j]) / dx * dy
                
        # 5. Cálculo Final do Gradiente Adjunto (adjointgradient)
        # Combina a derivada direta (fc) com a regra da cadeia da matriz A e vetor bp
        adjoint_grad = np.zeros(len(fc), dtype=np.float32)
        
        for i in range(len(fc)):
            if i in sim.Ap and i in sim.bp:
                # λ' * ( -Ap * T + bp )
                # No SciPy, sim.Ap[i] é a matriz derivada em relação a c_i
                term = -sim.Ap[i].dot(T_flat) + sim.bp[i]
                adjoint_grad[i] = fc[i] + np.dot(lambda_vec, term)
            else:
                adjoint_grad[i] = fc[i]
                
        # 6. Converter para tensor PyTorch e aplicar a regra da cadeia final (grad_output)
        grad_tensor = torch.tensor(adjoint_grad, dtype=torch.float32, device=grad_output.device)
        
        final_grad = grad_output * grad_tensor
        
        # Devolve o gradiente em relação a 'c_tensor'.
        # O segundo retorno é None porque o argumento 'sim' não requer gradiente (são parâmetros fixos).
        return final_grad, None