import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Tuple
from src.objects import NNstruct

class LambdaLayer(nn.Module):
    """
    Camada auxiliar para injetar funções (lambdas) e ativações 
    dentro de um nn.Sequential do PyTorch.
    """
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)

def build_dense_block(in_dims: List[int], out_dims: List[int], funcs: List[Callable]) -> nn.Sequential:
    """
    Equivalente à função `layerlist` do Julia.
    Constrói a lista de camadas lineares densas (Dense) pareadas com suas ativações.
    """
    layers = []
    for u, v, w in zip(in_dims, out_dims, funcs):
        layers.append(nn.Linear(u, v))
        layers.append(LambdaLayer(w))
    return nn.Sequential(*layers)

class PEDSModel(nn.Module):
    """
    Equivalente à função `initmodel`.
    Encapsula o Gerador, a Rede de Variância e o Peso de Combinação.
    """
    def __init__(self, nn_struct: NNstruct):
        super().__init__()
        
        # --- Modelo Gerador (Generator) ---
        gen_layers = []
        gen_layers.append(build_dense_block(nn_struct.inGen, nn_struct.outGen, nn_struct.funGen))
        for post_fn in nn_struct.postGen:
            gen_layers.append(LambdaLayer(post_fn))
        self.mgen = nn.Sequential(*gen_layers)
        
        # --- Peso de Combinação (Combining Weight) ---
        # Definido como nn.Parameter para garantir que, se for treinável no futuro,
        # o PyTorch consiga rastrear o gradiente. Inicializado como 0.5.
        self.cw = nn.Parameter(torch.tensor([0.5]))
        
        # --- Modelo de Variância (Variance) ---
        var_layers = []
        var_layers.append(LambdaLayer(nn_struct.preVar))
        var_layers.append(build_dense_block(nn_struct.inVar, nn_struct.outVar, nn_struct.funVar))
        self.mvar = nn.Sequential(*var_layers)

    def forward(self, *args):
        # O forward não é invocado diretamente de forma simples no PEDS
        # pois o fluxo de dados intercala o simulador físico.
        raise NotImplementedError("Use os submódulos .mgen e .mvar diretamente no loop de treinamento PEDS.")

class BaselineModel(nn.Module):
    """
    Equivalente à função `initbase`.
    O modelo base (Data-Driven puro) para comparar a eficácia do PEDS (Física-Guiada).
    """
    def __init__(self, nn_struct: NNstruct):
        super().__init__()
        
        # --- Modelo Gerador Base ---
        self.mgen = build_dense_block(nn_struct.inGen, nn_struct.outGen, nn_struct.funGen)
        
        # --- Modelo de Predição ---
        pred_layers = [
            nn.Linear(nn_struct.outGen[-1], 2),
            LambdaLayer(torch.tanh)
        ]
        for post_fn in nn_struct.postBase:
            pred_layers.append(LambdaLayer(post_fn))
        self.pred = nn.Sequential(*pred_layers)
        
        # --- Modelo de Variância Base ---
        self.mvar = nn.Sequential(
            nn.Linear(nn_struct.outGen[-1], 1),
            LambdaLayer(F.softplus)
        )

# ==========================================
# Funções de Avaliação de Ensemble
# ==========================================

def ensmean(mu_s: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Calcula a média do ensemble ao longo de uma dimensão específica."""
    return torch.mean(mu_s, dim=dim)

def ensvar(mu1s: torch.Tensor, mu2s: torch.Tensor, sigmas: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Equivalente a `ensvar`.
    Calcula a variância combinada do ensemble utilizando as predições e incertezas.
    """
    ens_mean_sq = ensmean(mu1s, dim)**2 + ensmean(mu2s, dim)**2
    # Equivalente ao @. (broadcasting implícito) do Julia
    combined = sigmas**2 + (mu1s**2 + mu2s**2 - ens_mean_sq)
    return torch.mean(combined, dim=dim)