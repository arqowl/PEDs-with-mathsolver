# src/physics/fourier_solver.py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter

def sdiff1(x: np.ndarray) -> sp.csr_matrix:
    """
    Computes the first-derivative finite-difference matrix 
    for Dirichlet boundaries (function = 0 at endpoints).
    """
    N = len(x) - 2
    # Forward differences: 1 / (x_{i+1} - x_i)
    dx1 = 1.0 / (x[1:N+1] - x[0:N])
    dx2 = -1.0 / (x[2:N+2] - x[1:N+1])
    
    # Create sparse matrix with diagonals at offset 0 and -1
    D = sp.diags([dx1, dx2], [0, -1], shape=(N + 1, N))
    return D.tocsr()

def sdiff1_periodic(x: np.ndarray) -> sp.csr_matrix:
    """
    First-derivative matrix with periodic boundary conditions.
    """
    N = len(x) - 1 # Ajustado para N ser o tamanho exato do grid
    
    # Diferenças centrais ou forward consistentes
    dx = 1.0 / (x[1] - x[0])
    
    # Criar diagonais principais
    main_diag = np.ones(N) * dx
    lower_diag = np.ones(N-1) * -dx
    
    # Construir matriz com wrap-around (periódica)
    data = [main_diag, lower_diag]
    offsets = [0, -1]
    
    D_lil = sp.diags(data, offsets, shape=(N, N)).tolil()
    
    # Fechar o ciclo periódico (o último elemento aponta para o primeiro)
    D_lil[0, -1] = -dx 
    D_lil[-1, 0] = dx
    
    return D_lil.tocsr()

def get_position(L: float, resolution: int) -> np.ndarray:
    """Creates the spatial grid."""
    nx = int(round(L * resolution))
    delta = 1.0 / resolution
    # Note: Julia is 1-indexed (1:nx), NumPy is 0-indexed.
    # We maintain the physical coordinates.
    return np.arange(1, nx + 1) * delta


def laplacian(x: np.ndarray, y: np.ndarray, c: np.ndarray, periodicy: bool = True, arrayC: bool = True) -> sp.csc_matrix:
    Dx = sdiff1(x)
    Nx = Dx.shape[1]
    Dy = sdiff1_periodic(y) if periodicy else sdiff1(y)
    Ny = Dy.shape[1]
    
    # Operador Gradiente G: [Nx*Ny, Nx*Ny] -> [2*Nx*Ny, Nx*Ny]
    G = sp.vstack([sp.kron(sp.eye(Ny), Dx), sp.kron(Dy, sp.eye(Nx))])
    N_rows = G.shape[0]

    if arrayC:
        # Se 'c' vem da rede neural, ele deve ter tamanho igual ao número de elementos
        # em G (ou ser mapeável para isso).
        c_flat = np.array(c).flatten()
        if len(c_flat) != N_rows:
            # Resampling para casar com a física do solver
            c_expanded = np.interp(np.linspace(0, 1, N_rows), np.linspace(0, 1, len(c_flat)), c_flat)
        else:
            c_expanded = c_flat
        C = sp.diags(c_expanded, offsets=0, shape=(N_rows, N_rows), format='csc')
    else:
        # Lógica para c como função chamável
        xp, yp = 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])
        if periodicy:
            X1, Y1 = np.meshgrid(xp, y[:-1], indexing='ij')
            X2, Y2 = np.meshgrid(x[1:-1], yp, indexing='ij')
        else:
            X1, Y1 = np.meshgrid(xp, y[1:-1], indexing='ij')
            X2, Y2 = np.meshgrid(x[1:-1], yp, indexing='ij')
        
        C = sp.diags(np.concatenate([c(X1, Y1).flatten('F'), c(X2, Y2).flatten('F')]), 
                     offsets=0, shape=(N_rows, N_rows), format='csc')
            
    return (-G.T @ C @ G).tocsc()