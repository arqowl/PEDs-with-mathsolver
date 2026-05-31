import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

# Importações da nossa topologia
from src.objects import CSstruct, NNstruct, ALstruct, DataRunner, DataSet
from src.models.surrogate import PEDSModel
from src.models.losses import calculate_nll_loss, sync_tensor_across_gpus, varfilter
from src.data.loaders import initloader, getloader

def setup_distributed():
    """
    Inicializa o ambiente Distributed Data Parallel (DDP).
    O torchrun tratará das variáveis de ambiente automaticamente.
    """
    dist.init_process_group(backend="nccl") # nccl é otimizado para GPUs NVIDIA
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, device

def cleanup_distributed():
    """Encerra os processos de comunicação de forma limpa."""
    dist.destroy_process_group()

def train_active_learning():
    """
    O Loop Principal de Treino com Active Learning (AL).
    """
    # 1. Configuração de Hardware e DDP
    local_rank, device = setup_distributed()
    is_leader = (local_rank == 0) # O Rank 0 é o "líder", responsável por imprimir logs

    if is_leader:
        print("[PEDS] A iniciar a Arquitetura de Treino Distribuído...")

    # 2. Carregar Hiperparâmetros
    cs = CSstruct()
    nn_params = NNstruct()
    al = ALstruct()

    # 3. Inicializar Modelo e mapear para a GPU correta
    model = PEDSModel(nn_params).to(device)
    
    # Envolver o modelo no DDP para sincronização automática de gradientes
    model = DDP(model, device_ids=[local_rank])
    
    # 4. Configurar Otimizador (O Julia usa Adam ou Flux.Optimise implicitamente)
    optimizer = Adam(model.parameters(), lr=al.eta)

    # 5. Inicializar Estruturas de Dados (Mockups para o esqueleto)
    # Na prática, você carregará os seus X_valid.csv e Y_valid.csv aqui
    # dr = DataRunner(X_total, Y_total, start=0)
    # ds = DataSet()
    # train_loader = initloader(al, dr, ds)
    
    if is_leader:
        print("[PEDS] A iniciar as Iterações de Active Learning...")

    # 6. O Loop de Active Learning (AL)
    for al_step in range(al.J): # al.J é o número de iterações de Active Learning
        if is_leader:
            print(f"\n--- Fase de Active Learning {al_step + 1}/{al.J} ---")

        # 6.1. O Loop de Épocas (Treino da Rede Neural)
        for epoch in range(al.T): # al.T é o número de épocas por ciclo de AL
            model.train()
            epoch_loss = torch.tensor(0.0, device=device)
            
            # --- LOOP DO BATCH ---
            # for batch_idx, (X_batch, y_target) in enumerate(train_loader):
            # Substitua este bloco quando tiver os dados carregados:
            # X_batch, y_target = X_batch.to(device), y_target.to(device)
            
            # (MOCKUP PARA O ESQUELETO RODAR SEM DADOS REAIS)
            X_batch = torch.randn((al.batchsize, 13), device=device) # Features de entrada
            y_target = torch.complex(torch.randn(al.batchsize), torch.randn(al.batchsize)).to(device)

            optimizer.zero_grad()

            # --- O FLUXO HÍBRIDO (Física + IA) ---
            # No PEDS original, mloglik devolve rp, ip, vp. 
            # Aqui demonstramos como a física seria acoplada:
            
            # Passo A: O Gerador (mgen) produz os parâmetros geométricos
            # geometry_params = model.module.mgen(X_batch)
            
            # Passo B: (Opcional) A Física (Solver de Baixa Fidelidade) atua sobre a geometria
            # ez_field = solver_fisico.solve(geometry_params)
            
            # Passo C: Extraímos rp e ip (respostas) e vp (variância do modelo mvar)
            # Para manter o esqueleto a funcionar, vamos simular a resposta de mloglik:
            rp = torch.randn(al.batchsize, device=device, requires_grad=True)
            ip = torch.randn(al.batchsize, device=device, requires_grad=True)
            vp = torch.abs(torch.randn(al.batchsize, device=device, requires_grad=True)) # Variância deve ser positiva

            outputs = (rp, ip, vp)

            # --- CÁLCULO DE PERDAS E GRADIENTES ---
            loss = calculate_nll_loss(outputs, y_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()

            # Sincroniza a perda média de todas as GPUs
            avg_epoch_loss = sync_tensor_across_gpus(epoch_loss) # / len(train_loader)

            if is_leader and (epoch % 2 == 0 or epoch == al.T - 1):
                print(f"Época {epoch+1}/{al.T} | Loss NLL: {avg_epoch_loss.item():.4f}")

        # 6.2. Filtro de Variância (Active Learning Step)
        if al_step < al.J - 1:
            if is_leader:
                print("[Active Learning] A procurar os dados com maior incerteza (varfilter)...")
            # Aqui chamaria o getloader e o varfilter (como traduzimos no loaders.py)
            # X_pool_amostra = dr.pegar_candidatos()
            # incertezas = varfilter(model, X_pool_amostra)
            # train_loader = getloader(al, dr, ds, filterfun)
            pass

    if is_leader:
        print("\n[PEDS] Treino Concluído com Sucesso! A guardar pesos...")
        # torch.save(model.module.state_dict(), "peds_weights.pth")

    cleanup_distributed()

if __name__ == "__main__":
    train_active_learning()