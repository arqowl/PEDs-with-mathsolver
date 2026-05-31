# Arquivo: src/__init__.py
# Equivalente arquitetural ao módulo PEDS.jl

import os
import random
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.distributed as dist

# ---------------------------------------------------------
# 1. Configuração de Precisão e Dispositivo (ChangePrecision / CUDA)
# ---------------------------------------------------------
# Define a precisão padrão para tensores (equivalente ao ChangePrecision no Julia)
# Fundamental para estabilidade em equações diferenciais
torch.set_default_dtype(torch.float32) 

# Identificação de hardware (CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PEDS Init] Aceleração de Hardware configurada para: {device}")

# ---------------------------------------------------------
# 2. Inicialização Distribuída (Equivalente ao MPI.Init())
# ---------------------------------------------------------
def init_distributed_processing():
    """
    Inicializa o processamento distribuído se o ambiente estiver configurado.
    Substitui o uso rígido do MPI para aproveitar o ecossistema PyTorch.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"[PEDS Init] Processamento Distribuído iniciado. Rank Local: {local_rank}")
    else:
        print("[PEDS Init] Executando em modo Single-Process / Local.")

# Executa a inicialização ao importar o pacote
init_distributed_processing()

# ---------------------------------------------------------
# 3. Mapeamento de Módulos (Equivalente aos "include(...)")
# ---------------------------------------------------------
# Na estrutura Python, as funcionalidades são importadas dos subdiretórios
# que criamos anteriormente. Exemplo de como a API será exposta:

# from .data.loaders import * # Equivalente a data.jl
# from .physics.diffusion import * # Equivalente a coarse.jl (baixa fidelidade)
# from .models.surrogate import * # Equivalente a models.jl e objects.jl

__all__ = ["device", "init_distributed_processing"]