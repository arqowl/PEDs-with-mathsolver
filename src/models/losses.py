import torch
import torch.nn.functional as F
import torch.distributed as dist

def sync_tensor_across_gpus(t: torch.Tensor) -> torch.Tensor:
    # Equivalente arquitetural a sum_reduce(comm, localsum) do Julia.
    # No PyTorch, os gradientes sao sincronizados pelo DDP automaticamente,
    # mas para metricas globais (como NLL ou Validacao), usamos dist.all_reduce.
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        # Dividimos pelo numero total de processos para ter a media
        t = t / dist.get_world_size()
    return t

def calculate_nll_loss(outputs: tuple, target: torch.Tensor) -> torch.Tensor:
    # Equivalente ao dNLL (Negative Log-Likelihood) do Julia.
    # outputs = (rp, ip, vp) onde vp e a variancia predita.
    rp, ip, vp = outputs
    # vp += 1e-6 (para evitar divisao por zero, como no Julia)
    vp = vp + 1e-6 
    # Calculo das componentes reais e imaginarias
    real_diff_sq = (rp - target.real)**2
    imag_diff_sq = (ip - target.imag)**2
        # sadd = log(vp) + ((rp-real(y[i]))^2 + (ip-imag(y[i]))^2)/2/vp^2
    sadd = torch.log(vp) + (real_diff_sq + imag_diff_sq) / (2 * (vp**2))
    # O DDP se encarrega de sincronizar a media disso durante o backward
    return torch.mean(sadd)

def varfilter(model, X_pool: torch.Tensor) -> torch.Tensor:
    # Equivalente a funcao varfilter. 
    # Avalia os modelos sobre um pool de dados (Active Learning) e 
    # retorna a incerteza (variancia combinada) de cada ponto.
    model.eval()
    with torch.no_grad():
        # Retorna as predicoes reais, imaginarias e a variancia
        # Para um unico modelo:
        rp, ip, vp = model(X_pool)
        uncertainties = vp.squeeze()
    return uncertainties