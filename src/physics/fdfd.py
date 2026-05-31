import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from src.physics.geometry import SimulationDomain

class FDFDSolver:
    """
    Motor físico de baixa fidelidade para resolver as Equações de Maxwell 2D (FDFD).
    Traduz a lógica do coarse.jl usando matrizes esparsas do SciPy.
    """
    def __init__(self, sd: SimulationDomain, Rpml: float = 1e-20):
        self.sd = sd
        self.Rpml = Rpml

    def build_maxwell_operator(self, geometry: np.ndarray) -> sp.csr_matrix:
        """
        Constrói a matriz A gigante (Laplaciano 2D + Geometria) para resolver Ax = b.
        Equivalente à função `Maxwell_2d` no Julia.
        """
        nx, ny = self.sd.nx, self.sd.ny
        npml = self.sd.npml
        delta = 1.0 / self.sd.resolution
        omega = self.sd.omega

        # --- Operador Laplaciano em X ---
        o_x = np.ones(nx) / delta
        # Cria matriz de diferenças finitas D em x (shape: nx+1, nx)
        D_x = sp.diags([-o_x, o_x], [-1, 0], shape=(nx + 1, nx))
        lap_x = (D_x.T @ D_x).tolil() # tolil() permite alterar índices específicos com eficiência
        
        # Condição de contorno periódica em X
        lap_x[-1, 0] -= 1.0 / (delta ** 2)
        lap_x[0, -1] -= 1.0 / (delta ** 2)
        lap_x = lap_x.tocsr()

        # --- Operador Laplaciano em Y (com PML) ---
        o_y = np.ones(ny) / delta
        y = self.sd.ys
        y_prime = y + (0.5 * delta) # Derivada intercalada na malha

        # Perfil de absorção PML
        sigma0 = -np.log(self.Rpml) / (4.0 * (self.sd.dpml ** 3) / 3.0)
        
        # Função para calcular sigma dependendo da posição na malha
        def calc_sigma(xi):
            return np.where(xi > self.sd.Ly, sigma0 * (xi - self.sd.Ly)**2, 
                   np.where(xi < 0, sigma0 * (xi)**2, 0.0))

        sigma = calc_sigma(y)
        sigma_prime = calc_sigma(y_prime)

        # Matrizes diagonais de atenuação do PML
        Sigma = sp.diags(1.0 / (1.0 + (1j / omega) * sigma))
        Sigma_prime = sp.diags(1.0 / (1.0 + (1j / omega) * sigma_prime))

        D_y = sp.diags([-o_y, o_y], [-1, 0], shape=(ny + 1, ny))
        lap_y = Sigma @ D_y.T @ Sigma_prime @ D_y

        # --- Laplaciano 2D (Produto de Kronecker) ---
        Ix = sp.eye(nx)
        Iy = sp.eye(ny)
        lap_2d = sp.kron(Ix, lap_y) + sp.kron(lap_x, Iy)

        # --- Adição da Geometria (Permissividade) ---
        if geometry.size != nx * ny:
            raise ValueError(f"Geometria deve ter formato ({ny}, {nx})")
            
        geom_flat = (omega ** 2 * geometry).flatten()
        geom_diag = sp.diags(geom_flat)

        A = lap_2d - geom_diag
        return A.tocsc() # CSC é o formato mais rápido para solver de sistemas lineares no SciPy

    def get_continuous_source(self) -> np.ndarray:
        """
        Gera o vetor b (fonte contínua) posicionado corretamente no domínio.
        """
        J = np.zeros((self.sd.ny, self.sd.nx), dtype=np.complex128)
        
        # Em Python, indexamos do topo/fim de forma diferente do Julia (end - ...)
        # O Julia arredonda, vamos manter a mesma lógica
        source_idx = int(round(self.sd.ny - (self.sd.dpml + self.sd.source) * self.sd.resolution))
        J[source_idx, :] = 1j * self.sd.omega * self.sd.resolution
        
        return J.flatten()

    def get_monitor_mask(self) -> np.ndarray:
        """
        Gera a máscara booleana onde a transmissão complexa será medida.
        """
        M = np.zeros((self.sd.ny, self.sd.nx), dtype=bool)
        monitor_idx = int(round(self.sd.ny - (self.sd.dpml + self.sd.monitor) * self.sd.resolution))
        M[monitor_idx, :] = True
        return M

    def solve_em_field(self, geometry: np.ndarray) -> np.ndarray:
        """
        Resolve A * Ez = J para encontrar o campo eletromagnético.
        """
        A = self.build_maxwell_operator(geometry)
        b = self.get_continuous_source()
        
        # spla.spsolve é o equivalente exato ao operador `A \ b` do Julia
        Ez_flat = spla.spsolve(A, b)
        return Ez_flat.reshape((self.sd.ny, self.sd.nx))

    def complex_transmission(self, Ez: np.ndarray) -> complex:
        """
        Calcula a transmissão média no ponto de monitoramento.
        """
        mask = self.get_monitor_mask()
        return np.mean(Ez[mask])