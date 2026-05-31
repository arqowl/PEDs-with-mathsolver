import numpy as np
from abc import ABC, abstractmethod

class LowFidelitySimulator(ABC):
    """
    Interface base para os simuladores físicos de baixa fidelidade.
    Garante o contrato para qualquer PDE que formos modelar.
    """
    @abstractmethod
    def solve(self, initial_conditions: np.ndarray, steps: int) -> np.ndarray:
        pass

class DiffusionSolver(LowFidelitySimulator):
    """
    Implementação da equação de difusão (ex: propagação térmica).
    Aqui traduziremos a lógica do solver de diferenças finitas do Julia.
    """
    def __init__(self, alpha: float, dx: float, dt: float):
        self.alpha = alpha  # Coeficiente de difusão
        self.dx = dx        # Espaçamento da malha espacial
        self.dt = dt        # Passo de tempo
        
        # Validação de estabilidade de Courant-Friedrichs-Lewy (CFL)
        if (self.alpha * self.dt) / (self.dx ** 2) > 0.5:
            raise ValueError("Parâmetros instáveis para o método explícito.")

    def solve(self, initial_conditions: np.ndarray, steps: int) -> np.ndarray:
        # O estado atual da malha
        u = initial_conditions.copy()
        
        # Aqui entrará a tradução do loop de tempo e espaço do Julia
        # substituindo loops for pesados por operações vetorizadas do NumPy
        
        return u