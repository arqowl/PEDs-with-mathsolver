from dataclasses import dataclass, field
from typing import List, Callable, Any
import torch
import torch.nn.functional as F

@dataclass
class NNstruct:
    """
    Parâmetros da arquitetura da Rede Neural (Gerador e Variância).
    Usamos field(default_factory=...) para evitar que listas mutáveis 
    sejam compartilhadas entre diferentes instâncias (um erro comum em Python).
    """
    # Parâmetros da rede geradora (Generator)
    inGen: List[int] = field(default_factory=lambda: [13, 256, 256])
    outGen: List[int] = field(default_factory=lambda: [256, 256, 19 * 221])
    
    # Funções de ativação do PyTorch
    funGen: List[Callable] = field(default_factory=lambda: [F.relu, F.relu, F.hardtanh])
    
    # Processamento pós-gerador (Lambdas)
    # Nota de Arquitetura: Em Julia a ordem do reshape é (221, 19, Batch). 
    # Em PyTorch a convenção é (Batch, H, W). Ajustado para (Batch, 221, 19).
    postGen: List[Callable] = field(default_factory=lambda: [
        lambda x: x * 1.5 + 2.5,
        lambda x: x.reshape(-1, 221, 19) 
    ])
    
    # Fator multiplicador para combinação
    multfact: float = 100.0
    
    # Parâmetros da rede de variância
    preVar: Callable = lambda x: x.reshape(x.shape[0], -1) # Mantém o Batch isolado (Batch, Features)
    inVar: List[int] = field(default_factory=lambda: [4199, 256, 256, 256])
    outVar: List[int] = field(default_factory=lambda: [256, 256, 256, 1])
    funVar: List[Callable] = field(default_factory=lambda: [F.relu, F.relu, F.relu, F.relu])
    
    # Baseline
    postBase: List[Callable] = field(default_factory=lambda: [lambda x: 1.3 * x])

@dataclass
class CSstruct:
    """
    Parâmetros do Coarse Solver (Simulador Físico de Baixa Fidelidade).
    Esta é a classe que passaremos para instanciar o SimulationDomain.
    """
    resolution: int = 20
    Lx: float = 0.95
    Ly: float = 17.0
    dpml: float = 2.0
    source: float = 1.0
    monitor: float = 16.0
    epssub: float = 1.45 ** 2
    refracsim: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.45])
    ny_nn: int = 221 
    nn_x: int = 19
    interstice: float = 0.5
    hole: float = 0.75
    
    # Transmissão de referência (em Python o imaginário é 'j' ao invés de 'im')
    refsim: complex = -0.33133778612182957 + 0.12500380630233138j

@dataclass
class ALstruct:
    """
    Hiperparâmetros do Active Learning (Treinamento).
    """
    J: int = 5
    T: int = 8
    Ninit: int = 256
    K: int = 64
    M: int = 4
    ne: int = 10
    batchsize: int = 64
    eta: float = 1e-3  # Taxa de aprendizado (learning rate)
    Nvalid: int = 512

@dataclass
class DataRunner:
    """Estrutura auxiliar para iteração de dados"""
    X: Any
    y: Any
    start: int

@dataclass
class DataSet:
    """Gerenciador de estado do conjunto de dados"""
    X: list = field(default_factory=list)
    y: list = field(default_factory=list)

# Instância global para ser importada pelos outros módulos (equivalente ao cs no Julia)
cs = CSstruct()