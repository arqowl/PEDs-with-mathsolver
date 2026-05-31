import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from typing import Callable, Tuple

# Importando as estruturas que criamos no src/objects.py
from src.objects import ALstruct, DataRunner, DataSet

def parse_julia_complex(val: str) -> complex:
    """
    Parser robusto para ler números complexos gerados em Julia.
    Converte strings como "-0.33133 + 0.12500im" para objetos complexos nativos do Python.
    """
    if pd.isna(val):
        return 0j
    # Remove espaços e troca 'im' por 'j'
    val_clean = str(val).replace(' ', '').replace('im', 'j')
    return complex(val_clean)

def take(dr: DataRunner, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extrai um lote (batch) sequencial de tamanho 'n' do DataRunner.
    Adaptação Arquitetural: Assume que dr.X e dr.y já estão no padrão PyTorch (Batch, ...).
    """
    s = dr.start
    if s + n > len(dr.y):
        raise ValueError(f"n = {n} excede o número de pontos restantes no DataRunner ({len(dr.y) - s}).")
    
    dr.start += n
    
    # Slicing focado na dimensão do batch (índice 0)
    X_batch = dr.X[s : s + n]
    y_batch = dr.y[s : s + n]
    
    return X_batch, y_batch

def initloader(al: ALstruct, dr: DataRunner, ds: DataSet) -> DataLoader:
    """
    Inicializa o DataLoader para a fase de pré-treinamento com os dados iniciais (Ninit).
    Equivalente ao `initloader` do Julia.
    """
    X, y = take(dr, al.Ninit)
    
    # Atualiza o estado global do dataset
    ds.X = X
    ds.y = y
    
    dataset = TensorDataset(ds.X, ds.y)
    return DataLoader(dataset, batch_size=al.batchsize, shuffle=True)

def initvalid(al: ALstruct, drv: DataRunner) -> DataLoader:
    """
    Inicializa o DataLoader de validação a partir de um DataRunner separado.
    """
    X, y = take(drv, al.Nvalid)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=al.batchsize, shuffle=False)

def getloader(al: ALstruct, dr: DataRunner, ds: DataSet, filterfun: Callable[[torch.Tensor], torch.Tensor]) -> DataLoader:
    """
    O Coração do Active Learning (AL).
    Mostra 'K * M' exemplos para a rede de variância (filterfun), escolhe os 'K'
    exemplos com maior incerteza e os adiciona ao dataset de treinamento.
    """
    # 1. Pega uma piscina maior de candidatos
    X_sampled, y_sampled = take(dr, al.K * al.M)
    
    # 2. Avalia a incerteza/variância usando a função fornecida (modelo de IA)
    # A função deve retornar um tensor 1D de scores.
    with torch.no_grad():
        scores = filterfun(X_sampled)
    
    # 3. Ordena e pega os índices dos 'K' maiores valores (sortperm do Julia)
    # torch.argsort retorna do menor pro maior, então pegamos os últimos 'K'
    top_k_indices = torch.argsort(scores)[-al.K :]
    
    X_selected = X_sampled[top_k_indices]
    y_selected = y_sampled[top_k_indices]
    
    # 4. Concatena os novos exemplos ao dataset global (Equivalente ao hcat/vcat)
    ds.X = torch.cat([ds.X, X_selected], dim=0)
    ds.y = torch.cat([ds.y, y_selected], dim=0)
    
    # 5. Retorna o novo DataLoader atualizado
    dataset = TensorDataset(ds.X, ds.y)
    return DataLoader(dataset, batch_size=al.batchsize, shuffle=True)

def validationloader(al: ALstruct, validpath: str) -> DataLoader:
    """
    Carrega o dataset de validação diretamente de arquivos CSV do disco.
    Lida explicitamente com o parsing dos complexos do Julia.
    """
    x_path = os.path.join(validpath, "X_valid.csv")
    y_path = os.path.join(validpath, "y_valid.csv")
    
    # Carrega X (assumindo que são floats reais)
    # O arquivo gerado pelo Julia pode ser delimitado por vírgula e sem header
    X_np = pd.read_csv(x_path, header=None).values
    
    # Carrega y (assumindo que são strings complexas do Julia)
    df_y = pd.read_csv(y_path, header=None, dtype=str)
    # Aplica o parser em todos os elementos e converte para complex128
    y_np = df_y.applymap(parse_julia_complex).values.flatten()
    
    # Converte para Tensores PyTorch garantindo as precisões corretas
    # (float32 para a rede neural processar rápido e complex64 para as respostas da física)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.complex64)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=al.batchsize, shuffle=False)