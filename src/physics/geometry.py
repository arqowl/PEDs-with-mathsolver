import numpy as np

class SimulationDomain:
    """
    Equivalente à `struct SimulationDomain` e suas funções auxiliares no Julia.
    Gerencia as dimensões espaciais e a resolução da malha.
    """
    def __init__(self, Lx: float, Ly: float, omega: float, dpml: float, resolution: int, source: float, monitor: float):
        if resolution <= 0:
            raise ValueError("A resolução deve ser um inteiro positivo.")
        if not (0 < source < Ly):
            raise ValueError("A fonte deve estar dentro do domínio computacional (0 < source < Ly).")
        if not (0 < monitor < Ly):
            raise ValueError("O monitor deve estar dentro do domínio computacional (0 < monitor < Ly).")
            
        self.Lx = Lx
        self.Ly = Ly
        self.omega = omega
        self.dpml = dpml
        self.resolution = resolution
        self.source = source
        self.monitor = monitor

    @property
    def nx(self) -> int:
        return int(round(self.Lx * self.resolution))

    @property
    def ny(self) -> int:
        return int(round((self.Ly + 2 * self.dpml) * self.resolution))

    @property
    def npml(self) -> int:
        return int(round(self.dpml * self.resolution))

    @property
    def xs(self) -> np.ndarray:
        return np.arange(1, self.nx + 1) / self.resolution

    @property
    def ys(self) -> np.ndarray:
        return np.arange(1 - self.npml, self.ny - self.npml + 1) / self.resolution

def epsilon_hole_layers(x: np.ndarray, y: np.ndarray, ps: np.ndarray, 
                        refractive_indexes: tuple = (0, 0, 0), 
                        interstice: float = 0.5, hole: float = 0.75) -> np.ndarray:
    """
    Equivalente a `ϵ_hole_layers`.
    Constrói a matriz de geometria (permissividade dielétrica) com os buracos.
    """
    assert len(x) > 2, "Tamanho de x deve ser maior que 2 para definir delta"
    
    nx, ny = len(x), len(y)
    delta = x[1] - x[0]
    Ly_pml = y[-1] - y[0] + delta
    Lx = x[-1] - x[0] + delta
    
    assert np.all(ps > delta), "A média de pixels precisa suportar todos os casos."
    assert np.all(ps <= Lx), "Os buracos não podem ser maiores que o período."

    # Propriedades do material da célula unitária
    if refractive_indexes == (0, 0, 0):
        refractive_index_background = 1.0
        refractive_index_hole = 1.0
        refractive_index_substrate = 1.45
    else:
        refractive_index_background, refractive_index_hole, refractive_index_substrate = refractive_indexes

    eps_background = refractive_index_background ** 2
    eps_hole = refractive_index_hole ** 2
    eps_substrate = refractive_index_substrate ** 2

    # Inicializa a geometria com o background
    geometry = np.ones((ny, nx), dtype=np.complex128) * eps_background

    # Define o substrato (80% do domínio é substrato no PEDS original)
    index_top_substrate = int(np.floor(Ly_pml * 0.35 / delta))
    geometry[index_top_substrate:, :] = eps_substrate

    # Tratamento de sub-pixel na grade
    w_offset = 0.5 if x[nx // 2] == 0 else 0.0

    # Construção dos buracos
    number_holes = len(ps)
    n_inter_hole = int(np.floor(interstice / refractive_index_substrate / delta))
    n_hole_height = int(np.floor(hole / delta))

    assert index_top_substrate + number_holes * (n_inter_hole + n_hole_height) < ny

    for it_holes in range(number_holes):
        half_width = ps[it_holes] / (2 * delta) - w_offset
        n_half_width = int(np.floor(half_width))
        weight_eps_hole = half_width - n_half_width

        # Dentro dos buracos
        n_start = int(np.floor((nx - 2 * n_half_width) / 2 - w_offset))
        
        y_start = index_top_substrate + it_holes * n_inter_hole + it_holes * n_hole_height
        y_end = y_start + n_inter_hole + n_hole_height
        
        x_start = n_start
        x_end = n_start + int(np.floor(2 * (n_half_width + w_offset)))
        
        # O Julia usa indexação 1-based, no Python ajustamos para 0-based
        geometry[y_start:y_end, x_start:x_end] = eps_hole

        # Média de pixels (Pixel averaging)
        # Esquerda
        geometry[y_start:y_end, x_start - 1] = weight_eps_hole * eps_hole + (1 - weight_eps_hole) * eps_substrate
        # Direita
        geometry[y_start:y_end, x_end] = weight_eps_hole * eps_hole + (1 - weight_eps_hole) * eps_substrate

    return geometry