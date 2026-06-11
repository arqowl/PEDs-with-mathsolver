# src/peds_experiments.py
# Biblioteca para rodar os experimentos PEDS de difusão (Fourier/Fisher),
# registrar curvas de aprendizado (baseline vs PEDS por época) e reproduzir
# as duas tabelas da página 21 do paper.
#
# Física VALIDADA: solver de difusão (adjunto ~1e-7) e generatepores
# (FE coarse Fourier(16)=0.140 vs paper 0.135). Para Fisher, ver CAVEAT abaixo.

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt  # backend padrão: renderiza inline no notebook
from scipy import stats
import json
import re

try:                                   # telemetria opcional (não quebra sem GPU/pynvml)
    from src.telemetria import MonitorTreinamento
    _HAS_TELE = True
except Exception:
    _HAS_TELE = False

from src.physics.diffusion_solver import (
    DiffusionSim, DifferentiableDiffusion, generatepores,
)

# Configuração de cada experimento (resolução coarse e arquivo de dados).
EXPERIMENTS = {
    "Fourier(16)": dict(data="fourier16", res=4),
    "Fourier(25)": dict(data="fourier25", res=5),
    "Fisher(16)":  dict(data="fisher16",  res=4),
    "Fisher(25)":  dict(data="fisher25",  res=5),
}

# Valores de referência do paper (página 21) para comparação.
PAPER = {
    "Fourier(16)": dict(peds=0.037, nn=0.051, lowfi=0.135),
    "Fourier(25)": dict(peds=0.038, nn=0.047, lowfi=0.085),
    "Fisher(16)":  dict(peds=0.045, nn=0.101, lowfi=0.381),
    "Fisher(25)":  dict(peds=0.055, nn=0.144, lowfi=0.367),
    "Maxwell(10)": dict(peds=0.19,  nn=0.56,  lowfi=1.24),
}


def fractional_error(pred, true):
    return (torch.linalg.norm(pred - true) / torch.linalg.norm(true)).item()


def huber(pred, true, delta=1e-3):
    return F.huber_loss(pred, true, delta=delta)


# ----------------------- modelos -----------------------
class _Lambda(nn.Module):
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, x): return self.fn(x)


def _generator(nin, nout, nnodes=128):
    # SI sec 7: 2 camadas ocultas 128 relu + dropout 0.5; saída res² hardtanh; pós-escala -> [0.1,1]
    return nn.Sequential(
        nn.Linear(nin, nnodes), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(nnodes, nnodes), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(nnodes, nout), _Lambda(lambda x: torch.clamp(x, -1, 1)),  # hardtanh
        _Lambda(lambda x: x * 0.9 / 2 + 0.45 + 0.1),
    )


class DiffusionPEDS(nn.Module):
    def __init__(self, res, sim, cw_init=0.05):
        super().__init__()
        self.res = res
        self.mgen = _generator(res * res, res * res)
        self.cw = nn.Parameter(torch.tensor([float(cw_init)]))
        self.solver = DifferentiableDiffusion(sim)

    def weight(self):
        return torch.clamp(self.cw, 0.0, 1.0)   # paper: w = clamp(cw,0,1)

    def forward(self, X):
        gen = self.mgen(X)                                  # (B, res²) in [0.1,1]
        coarse = 1.0 - 0.9 * X                              # generatepores(downsample), sem grad útil
        w = self.weight()
        geom = w * gen + (1.0 - w) * coarse
        return self.solver(geom)                            # fluxo κ (B,)


class DiffusionPEDSAffine(nn.Module):
    """Ablation do artigo: mesmo gerador/mistura do PEDS, mas sem solver físico.

    A saída é uma leitura afim da geometria combinada: kappa = a^T geom + b.
    Isso preserva o gargalo geométrico e a estrutura multi-fidelidade, removendo
    apenas a resolução física da EDP.
    """
    def __init__(self, res, cw_init=0.05):
        super().__init__()
        self.res = res
        nin = res * res
        self.mgen = _generator(nin, nin)
        self.cw = nn.Parameter(torch.tensor([float(cw_init)]))
        self.readout = nn.Linear(nin, 1)

    def weight(self):
        return torch.clamp(self.cw, 0.0, 1.0)

    def combined_geometry(self, X):
        gen = self.mgen(X)
        coarse = 1.0 - 0.9 * X
        w = self.weight()
        return w * gen + (1.0 - w) * coarse

    def forward(self, X):
        geom = self.combined_geometry(X)
        return self.readout(geom).squeeze(-1)


class DiffusionBaseline(nn.Module):
    # NN-only: gerador + camada totalmente conectada substituindo o solver.
    def __init__(self, res):
        super().__init__()
        nin = res * res
        self.net = nn.Sequential(
            nn.Linear(nin, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, nin), nn.ReLU(),
            nn.Linear(nin, 1), _Lambda(lambda x: torch.clamp(x, -1, 1)),
            _Lambda(lambda x: x * 0.9 / 2 + 0.45 + 0.1),
        )

    def forward(self, X):
        return self.net(X).squeeze(-1)


# ----------------------- dados -----------------------
def load_data(data_root, name, device):
    X = pd.read_csv(os.path.join(data_root, f"X_{name}_small.csv"), header=None).values.T  # (amostras, feat)
    y = pd.read_csv(os.path.join(data_root, f"y_{name}_small.csv"), header=None).values.flatten()
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    # Split fixo e sem vazamento: validação no primeiro bloco, teste no último,
    # treino no miolo. Com 10k pontos, o treino disponível é 7952 amostras.
    Xv, yv = X[:1024], y[:1024]
    Xtest, ytest = X[-1024:], y[-1024:]
    Xt, yt = X[1024:-1024], y[1024:-1024]
    return (Xt, yt), (Xv, yv), (Xtest, ytest)


# ----------------------- treino com tracking -----------------------
def _train_one(model, Xt, yt, Xtest, ytest, epochs, lr, batch, track_every,
               tele_name=None, return_preds=False):
    opt = Adam(model.parameters(), lr=lr)
    n = Xt.shape[0]
    hist_ep, hist_fe, hist_preds = [], [], []
    mon = None
    if tele_name and _HAS_TELE:
        mon = MonitorTreinamento(model); mon.modelo_name = tele_name
    for ep in range(epochs):
        if mon: mon.iniciar_epoca()
        model.train()
        perm = torch.randperm(n, device=Xt.device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = huber(pred, yt[idx])
            loss.backward()
            opt.step()
        if mon: mon.finalizar_epoca(ep + 1)
        if ep % track_every == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                p = model(Xtest)
            hist_ep.append(ep + 1); hist_fe.append(fractional_error(p, ytest))
            if return_preds:
                hist_preds.append(p.detach())
    if mon: mon.salvar_logs()
    if return_preds:
        return hist_ep, hist_fe, hist_preds
    return hist_ep, hist_fe


def _train_ensemble(make_model, K, Xt, yt, Xtest, ytest, epochs, lr, batch,
                    track_every, seed, ytest_ref, tele_prefix=None):
    """Treina K modelos (seeds distintos) e combina por época: a predição do
    ensemble é a MÉDIA das predições dos K (como no paper). Devolve a curva de
    FE do ensemble e o w médio (quando o modelo tem peso de mistura)."""
    preds_by_member, ws, ep_ref = [], [], None
    for k in range(K):
        torch.manual_seed(seed + k)
        m = make_model()
        tele = f"{tele_prefix}_m{k}" if tele_prefix else None
        ep, _, preds = _train_one(m, Xt, yt, Xtest, ytest, epochs, lr, batch,
                                  track_every, tele_name=tele, return_preds=True)
        ep_ref = ep; preds_by_member.append(preds)
        if hasattr(m, "weight"):
            ws.append(float(m.weight().item()))
    ens_fe = []
    for j in range(len(ep_ref)):
        mean_pred = torch.stack([preds_by_member[k][j] for k in range(K)]).mean(0)
        ens_fe.append(fractional_error(mean_pred, ytest_ref))
    w_mean = (sum(ws) / len(ws)) if ws else float("nan")
    return ep_ref, ens_fe, w_mean


def run_diffusion_experiment(name, data_root, device="cpu",
                             epochs=200, lr=5e-5, batch=64, ninit=1088,
                             track_every=5, seed=0, telemetry=False, n_ensemble=1):
    """Treina baseline e PEDS, devolve histórico de FE por época e FEs finais.
    n_ensemble>1: a predição é a média de n_ensemble modelos (como no paper)."""
    cfg = EXPERIMENTS[name]
    res = cfg["res"]
    sim = DiffusionSim(res)
    (Xt, yt), _, (Xtest, ytest) = load_data(data_root, cfg["data"], device)
    Xt, yt = Xt[:ninit], yt[:ninit]      # regime de poucos dados (~10³)
    tag = name.replace("(", "").replace(")", "")
    cw0 = 0.05 if "Fourier" in name else 0.45

    if n_ensemble == 1:
        torch.manual_seed(seed)
        peds = DiffusionPEDS(res, sim, cw_init=cw0).to(device)
        ep_p, fe_p = _train_one(peds, Xt, yt, Xtest, ytest, epochs, lr, batch, track_every,
                                tele_name=(f"{tag}_PEDS" if telemetry else None))
        w_val = float(peds.weight().item())
        torch.manual_seed(seed)
        base = DiffusionBaseline(res).to(device)
        ep_b, fe_b = _train_one(base, Xt, yt, Xtest, ytest, epochs, lr, batch, track_every,
                                tele_name=(f"{tag}_NNonly" if telemetry else None))
    else:
        ep_p, fe_p, w_val = _train_ensemble(
            lambda: DiffusionPEDS(res, sim, cw_init=cw0).to(device), n_ensemble,
            Xt, yt, Xtest, ytest, epochs, lr, batch, track_every, seed, ytest,
            tele_prefix=(f"{tag}_PEDS" if telemetry else None))
        ep_b, fe_b, _ = _train_ensemble(
            lambda: DiffusionBaseline(res).to(device), n_ensemble,
            Xt, yt, Xtest, ytest, epochs, lr, batch, track_every, seed, ytest,
            tele_prefix=(f"{tag}_NNonly" if telemetry else None))

    # FE low-fidelity (coarse puro, w=0) para a Tabela 3
    solver = DifferentiableDiffusion(sim).to(device)
    with torch.no_grad():
        lowfi = fractional_error(solver(1.0 - 0.9 * Xtest), ytest)

    return {
        "name": name,
        "epochs": ep_p, "fe_peds": fe_p, "fe_nn": fe_b,
        "final_peds": fe_p[-1], "final_nn": fe_b[-1], "lowfi": lowfi,
        "w": w_val, "n_ensemble": n_ensemble,
    }


# ----------------------- plots -----------------------
def plot_learning_curve(result, save_dir="./figs"):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result["epochs"], result["fe_nn"], "--", color="#1f77b4", marker="x", label="NN-only (baseline)")
    ax.plot(result["epochs"], result["fe_peds"], "-", color="#d62728", marker="o", ms=3, label="PEDS")
    if result["name"] in PAPER:
        ax.axhline(PAPER[result["name"]]["peds"], color="#d62728", ls=":", alpha=0.5,
                   label=f"PEDS paper ({PAPER[result['name']]['peds']:.3f})")
    ax.set_xlabel("época"); ax.set_ylabel("Fractional Error (teste)")
    ax.set_yscale("log"); ax.set_title(f"Curva de aprendizado — {result['name']}")
    ax.legend(); ax.grid(alpha=0.3)
    path = os.path.join(save_dir, f"learning_curve_{result['name'].replace('(','').replace(')','')}.png")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.show()
    return path


def plot_replication_tables(results, save_dir="./figs"):
    """Reproduz, no formato do artigo, apenas o que MEDIMOS de fato com os 10k pontos:
      Tabela 2: PEDS(10³) vs NN-only(10³)  — replicado e paper lado a lado.
                (sem colunas NN-only 10⁴/10⁵: os autores não disponibilizam >10k pontos)
      Tabela 3: PEDS(10³), Low-fidelity, Improvement
                (sem Speedup: exigiria o solver de alta-fidelidade para cronometrar)"""
    os.makedirs(save_dir, exist_ok=True)
    order = ["Fourier(16)", "Fourier(25)", "Fisher(16)", "Fisher(25)", "Maxwell(10)"]
    order = [n for n in order if n in results]
    ke = max((results[n].get("n_ensemble", 1) for n in order), default=1)
    tag = f"  [ensemble de {ke}]" if ke > 1 else "  [modelo único]"

    # Tabela 2: PEDS vs NN-only (10³), replicado e paper
    rows2 = [["Modelo", "PEDS (rep.)", "PEDS (paper)", "NN-only (rep.)", "NN-only (paper)"]]
    for n in order:
        r = results[n]; p = PAPER.get(n, {})
        rows2.append([n, f"{r['final_peds']*100:.1f}%", f"{p.get('peds',float('nan'))*100:.1f}%",
                      f"{r['final_nn']*100:.1f}%", f"{p.get('nn',float('nan'))*100:.1f}%"])

    # Tabela 3: PEDS vs low-fidelity + improvement
    rows3 = [["Modelo", "PEDS (rep.)", "Low-fi (rep.)", "Low-fi (paper)", "Improvement (rep.)"]]
    for n in order:
        r = results[n]; p = PAPER.get(n, {})
        imp = r["lowfi"] / r["final_peds"] if r["final_peds"] > 0 else float("nan")
        rows3.append([n, f"{r['final_peds']*100:.1f}%", f"{r['lowfi']*100:.1f}%",
                      f"{p.get('lowfi',float('nan'))*100:.1f}%", f"{imp:.1f}×"])

    paths = []
    for rows, title, fname in [
        (rows2, "Tabela 2 — PEDS vs NN-only, FE no teste (10³ pontos)" + tag, "tabela2_peds_vs_nn.png"),
        (rows3, "Tabela 3 — PEDS vs low-fidelity (10³ pontos)" + tag, "tabela3_peds_vs_lowfi.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 0.6 + 0.5 * len(rows)))
        ax.axis("off")
        tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
        for j in range(len(rows[0])):
            tbl[0, j].set_facecolor("#333"); tbl[0, j].set_text_props(color="white", fontweight="bold")
        ax.set_title(title, pad=14, fontweight="bold")
        path = os.path.join(save_dir, fname)
        fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.show()
        paths.append(path)
    return paths


# ----------------------- sweep de tamanho de treino (eficiência de dados) -----------------------
def run_size_sweep(name, data_root, sizes=(256, 512, 1024, 2048, 4096), device="cpu",
                   epochs=200, lr=5e-5, batch=64, seed=0, n_ensemble=1):
    """Treina PEDS e NN-only em vários tamanhos N de treino e devolve o FE final
    de cada um. É o análogo da Fig. 3 (FE × nº de pontos): mostra se o PEDS
    atinge um erro-alvo com muito menos dados que o baseline.
    n_ensemble>1: cada ponto é a média de n_ensemble modelos (suaviza as curvas)."""
    cfg = EXPERIMENTS[name]; res = cfg["res"]; sim = DiffusionSim(res)
    (Xt, yt), _, (Xtest, ytest) = load_data(data_root, cfg["data"], device)
    pool = Xt.shape[0]
    out = {"name": name, "sizes": [], "fe_peds": [], "fe_nn": [], "n_ensemble": n_ensemble}
    cw0 = 0.05 if "Fourier" in name else 0.45
    for N in sizes:
        if N > pool:
            print(f"  (pulando N={N}: pool de treino só tem {pool})"); continue
        XtN, ytN = Xt[:N], yt[:N]
        if n_ensemble == 1:
            torch.manual_seed(seed)
            peds = DiffusionPEDS(res, sim, cw_init=cw0).to(device)
            _, fp = _train_one(peds, XtN, ytN, Xtest, ytest, epochs, lr, batch, track_every=epochs)
            torch.manual_seed(seed)
            base = DiffusionBaseline(res).to(device)
            _, fb = _train_one(base, XtN, ytN, Xtest, ytest, epochs, lr, batch, track_every=epochs)
            fe_p, fe_n = fp[-1], fb[-1]
        else:                                   # média de n_ensemble modelos (como no paper)
            _, fpe, _ = _train_ensemble(lambda: DiffusionPEDS(res, sim, cw_init=cw0).to(device),
                                        n_ensemble, XtN, ytN, Xtest, ytest, epochs, lr, batch,
                                        epochs, seed, ytest)
            _, fbe, _ = _train_ensemble(lambda: DiffusionBaseline(res).to(device),
                                        n_ensemble, XtN, ytN, Xtest, ytest, epochs, lr, batch,
                                        epochs, seed, ytest)
            fe_p, fe_n = fpe[-1], fbe[-1]
        out["sizes"].append(N); out["fe_peds"].append(fe_p); out["fe_nn"].append(fe_n)
        print(f"  N={N:5d}: PEDS={fe_p:.3f}  NN-only={fe_n:.3f}")
    return out


def plot_size_sweep(sweep, target=0.05, save_dir="./figs"):
    """Plota FE × nº de pontos (loglog), PEDS vs NN-only, com a linha do erro-alvo."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(sweep["sizes"], sweep["fe_nn"], "--", color="#1f77b4", marker="x", label="NN-only")
    ax.loglog(sweep["sizes"], sweep["fe_peds"], "-", color="#d62728", marker="o", ms=4, label="PEDS")
    ax.axhline(target, color="gray", ls=":", label=f"erro-alvo {target*100:.0f}%")
    ax.set_xlabel("nº de pontos de treino"); ax.set_ylabel("Fractional Error (teste)")
    ke = sweep.get("n_ensemble", 1)
    suffix = f" (ensemble de {ke})" if ke > 1 else ""
    ax.set_title(f"Eficiência de dados — {sweep['name']}{suffix}")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    path = os.path.join(save_dir, f"size_sweep_{sweep['name'].replace('(','').replace(')','')}.png")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.show()
    return path


def plot_efficiency_table(sweeps, target=0.05, save_dir="./figs"):
    """Veredito do 'mais com menos': a partir dos sweeps, mostra com quantos pontos
    cada modelo atinge o erro-alvo e a economia de dados do PEDS sobre o NN-only."""
    os.makedirs(save_dir, exist_ok=True)
    rows = [["Modelo", f"N p/ PEDS≤{target*100:.0f}%", f"N p/ NN-only≤{target*100:.0f}%",
             "economia de dados (PEDS)"]]
    for sw in sweeps:
        def first_below(fes):
            for N, fe in zip(sw["sizes"], fes):
                if fe <= target:
                    return N
            return None
        npd, nnn = first_below(sw["fe_peds"]), first_below(sw["fe_nn"])
        nmax = sw["sizes"][-1]
        npd_s = str(npd) if npd else f"≥{nmax} (não atinge)"
        nnn_s = str(nnn) if nnn else f"≥{nmax} (não atinge)"
        if npd and nnn:
            econ = f"{nnn / npd:.1f}×"
        elif npd and not nnn:
            econ = f">{nmax / npd:.1f}× (NN não atinge)"
        else:
            econ = "—"
        rows.append([sw["name"], npd_s, nnn_s, econ])

    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    for j in range(len(rows[0])):
        tbl[0, j].set_facecolor("#333"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Eficiência de dados — quem atinge {target*100:.0f}% com menos pontos", pad=14, fontweight="bold")
    path = os.path.join(save_dir, "tabela_eficiencia_dados.png")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.show()
    return path


# ======================= Protocolo do artigo PEDS-afim =======================
ARTICLE_MODELS = ("peds", "peds_afim", "nn_only")
ARTICLE_LABELS = {
    "peds": "PEDS físico",
    "peds_afim": "PEDS-afim",
    "nn_only": "NN-only",
}
ARTICLE_COLORS = {
    "peds": "#d62728",
    "peds_afim": "#9467bd",
    "nn_only": "#1f77b4",
}


def _make_diffusion_model(kind, res, sim, cw_init):
    if kind == "peds":
        return DiffusionPEDS(res, sim, cw_init=cw_init)
    if kind == "peds_afim":
        return DiffusionPEDSAffine(res, cw_init=cw_init)
    if kind == "nn_only":
        return DiffusionBaseline(res)
    raise ValueError(f"Modelo desconhecido: {kind}")


def _predict_batches(model, X, batch=256):
    preds = []
    model.eval()
    with torch.no_grad():
        for s in range(0, X.shape[0], batch):
            preds.append(model(X[s:s + batch]).detach())
    return torch.cat(preds, dim=0)


def _fractional_error_batched(model, X, y, batch=256):
    pred = _predict_batches(model, X, batch=batch)
    return fractional_error(pred, y)


def _train_with_early_stopping(model, Xt, yt, Xv, yv, Xtest, ytest,
                               max_epochs=400, lr=5e-5, batch=64,
                               eval_every=10, patience=8, min_epochs=40,
                               min_delta=1e-4, eval_batch=256):
    """Treina uma semente usando validação fixa e restaura o melhor checkpoint.

    A paciência é contada em avaliações de validação, não em épocas. Isso mantém
    o custo sob controle porque o PEDS físico chama o solver coarse em cada eval.
    """
    opt = Adam(model.parameters(), lr=lr)
    n = Xt.shape[0]
    best_fe = float("inf")
    best_epoch = 0
    best_state = None
    checks_without_gain = 0
    history = []

    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=Xt.device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = huber(pred, yt[idx])
            loss.backward()
            opt.step()

        should_eval = (ep == 1) or (ep % eval_every == 0) or (ep == max_epochs)
        if not should_eval:
            continue

        val_fe = _fractional_error_batched(model, Xv, yv, batch=eval_batch)
        history.append((ep, val_fe))
        if val_fe < best_fe - min_delta:
            best_fe = val_fe
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checks_without_gain = 0
        else:
            checks_without_gain += 1

        if ep >= min_epochs and checks_without_gain >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(Xt.device) for k, v in best_state.items()})

    final_val_fe = _fractional_error_batched(model, Xv, yv, batch=eval_batch)
    test_fe = _fractional_error_batched(model, Xtest, ytest, batch=eval_batch)
    return {
        "test_fe": test_fe,
        "val_fe": final_val_fe,
        "best_epoch": best_epoch or ep,
        "epochs_run": ep,
        "history": history,
    }


def run_article_sweep(data_root, problems=None,
                      sizes=(64, 128, 256, 512, 1024, 2048, 4096),
                      seeds=range(10), models=ARTICLE_MODELS, device="cpu",
                      max_epochs=400, lr=5e-5, batch=64, eval_every=10,
                      patience=8, min_epochs=40, min_delta=1e-4,
                      eval_batch=256, save_dir="./results",
                      save_csv=True, resume=True, verbose=True):
    """Executa o protocolo do manuscrito para Fourier/Fisher.

    Retorna um DataFrame granular: uma linha por problema, modelo, N e semente.
    Essa tabela é a base para curvas, expoentes e testes não-paramétricos.
    """
    if problems is None:
        problems = list(EXPERIMENTS)
    seeds = list(seeds)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "article_seed_results.csv")
    rows = []
    completed = set()

    if resume and save_csv and os.path.exists(path):
        existing = pd.read_csv(path)
        key_cols = ["problem", "model", "N", "seed"]
        if all(c in existing.columns for c in key_cols):
            existing = existing.drop_duplicates(key_cols, keep="last")
            rows = existing.to_dict("records")
            completed = {
                (str(r["problem"]), str(r["model"]), int(r["N"]), int(r["seed"]))
                for r in rows
            }
            if verbose:
                print(f"Retomando de {path}: {len(completed)} combinações já concluídas.")

    for problem in problems:
        cfg = EXPERIMENTS[problem]
        res = cfg["res"]
        sim = DiffusionSim(res)
        (Xt_all, yt_all), (Xv, yv), (Xtest, ytest) = load_data(data_root, cfg["data"], device)
        pool = Xt_all.shape[0]
        cw0 = 0.05 if "Fourier" in problem else 0.45

        for N in sizes:
            if N > pool:
                if verbose:
                    print(f"[{problem}] pulando N={N}: treino seguro tem {pool} amostras")
                continue
            Xt, yt = Xt_all[:N], yt_all[:N]
            for kind in models:
                for seed in seeds:
                    key = (problem, kind, int(N), int(seed))
                    if key in completed:
                        if verbose:
                            print(f"{problem:11s} N={N:4d} seed={seed:2d} {kind:9s} skip=checkpoint")
                        continue
                    torch.manual_seed(int(seed))
                    np.random.seed(int(seed))
                    model = _make_diffusion_model(kind, res, sim, cw0).to(device)
                    info = _train_with_early_stopping(
                        model, Xt, yt, Xv, yv, Xtest, ytest,
                        max_epochs=max_epochs, lr=lr, batch=batch,
                        eval_every=eval_every, patience=patience,
                        min_epochs=min_epochs, min_delta=min_delta,
                        eval_batch=eval_batch,
                    )
                    w_val = float(model.weight().item()) if hasattr(model, "weight") else float("nan")
                    params = sum(p.numel() for p in model.parameters())
                    row = {
                        "problem": problem,
                        "data": cfg["data"],
                        "model": kind,
                        "model_label": ARTICLE_LABELS[kind],
                        "N": int(N),
                        "seed": int(seed),
                        "test_fe": info["test_fe"],
                        "val_fe": info["val_fe"],
                        "best_epoch": int(info["best_epoch"]),
                        "epochs_run": int(info["epochs_run"]),
                        "w": w_val,
                        "n_params": int(params),
                    }
                    rows.append(row)
                    completed.add(key)
                    if save_csv:
                        pd.DataFrame(rows).drop_duplicates(
                            ["problem", "model", "N", "seed"], keep="last"
                        ).to_csv(path, index=False)
                    if verbose:
                        print(
                            f"{problem:11s} N={N:4d} seed={seed:2d} "
                            f"{kind:9s} test={row['test_fe']:.4f} "
                            f"val={row['val_fe']:.4f} best_ep={row['best_epoch']}"
                        )

    df = pd.DataFrame(rows)
    if save_csv:
        df = df.drop_duplicates(["problem", "model", "N", "seed"], keep="last")
        df.to_csv(path, index=False)
        if verbose:
            print(f"Resultados granulares salvos em {path}")
    return df


def recover_article_results_from_notebook(notebook_path="notebooks/run_all_experiments.ipynb",
                                          save_dir="notebooks/results",
                                          output_name="article_seed_results.csv"):
    """Recupera linhas já impressas pelo notebook antes de uma interrupção.

    Útil quando a máquina desliga antes do fim do sweep. A recuperação usa apenas
    as linhas de stdout no formato emitido por run_article_sweep; campos que não
    eram impressos, como w e epochs_run, ficam como NaN.
    """
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)

    pattern = re.compile(
        r"^(?P<problem>Fourier\(\d+\)|Fisher\(\d+\))\s+"
        r"N=\s*(?P<N>\d+)\s+seed=\s*(?P<seed>\d+)\s+"
        r"(?P<model>peds_afim|nn_only|peds)\s+"
        r"test=(?P<test>[0-9.]+)\s+val=(?P<val>[0-9.]+)\s+best_ep=(?P<best>\d+)"
    )

    rows = []
    param_cache = {}
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            for line in str(text).splitlines():
                m = pattern.search(line.strip())
                if not m:
                    continue
                problem = m.group("problem")
                model = m.group("model")
                if (problem, model) not in param_cache:
                    cfg = EXPERIMENTS[problem]
                    sim = DiffusionSim(cfg["res"])
                    cw0 = 0.05 if "Fourier" in problem else 0.45
                    param_model = _make_diffusion_model(model, cfg["res"], sim, cw0)
                    param_cache[(problem, model)] = sum(p.numel() for p in param_model.parameters())
                rows.append({
                    "problem": problem,
                    "data": EXPERIMENTS[problem]["data"],
                    "model": model,
                    "model_label": ARTICLE_LABELS[model],
                    "N": int(m.group("N")),
                    "seed": int(m.group("seed")),
                    "test_fe": float(m.group("test")),
                    "val_fe": float(m.group("val")),
                    "best_epoch": int(m.group("best")),
                    "epochs_run": float("nan"),
                    "w": float("nan"),
                    "n_params": int(param_cache[(problem, model)]),
                })

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, output_name)
    if not df.empty:
        df = df.drop_duplicates(["problem", "model", "N", "seed"], keep="last")
        if os.path.exists(path):
            old = pd.read_csv(path)
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(["problem", "model", "N", "seed"], keep="last")
        df.to_csv(path, index=False)
    return df, path


def aggregate_article_results(df):
    def first_valid(x):
        valid = x.dropna()
        return valid.iloc[0] if len(valid) else np.nan

    grouped = df.groupby(["problem", "model", "model_label", "N"], as_index=False)
    return grouped.agg(
        n_seeds=("seed", "nunique"),
        mean_fe=("test_fe", "mean"),
        median_fe=("test_fe", "median"),
        std_fe=("test_fe", "std"),
        q25_fe=("test_fe", lambda x: x.quantile(0.25)),
        q75_fe=("test_fe", lambda x: x.quantile(0.75)),
        mean_best_epoch=("best_epoch", "mean"),
        mean_w=("w", "mean"),
        n_params=("n_params", first_valid),
    )


def fit_learning_exponents(df):
    """Ajusta FE ~ N^alpha por problema/modelo.

    Reporta alpha da curva de medianas e, quando possível, a distribuição de
    alpha por semente pareada ao longo dos N.
    """
    rows = []
    for (problem, model), sub in df.groupby(["problem", "model"]):
        med = sub.groupby("N")["test_fe"].median().sort_index()
        if len(med) < 2:
            continue
        alpha_med, intercept = np.polyfit(np.log(med.index.values), np.log(med.values), 1)

        seed_alphas = []
        for seed, ss in sub.groupby("seed"):
            curve = ss.groupby("N")["test_fe"].median().sort_index()
            if len(curve) == len(med):
                a, _ = np.polyfit(np.log(curve.index.values), np.log(curve.values), 1)
                seed_alphas.append(a)

        rows.append({
            "problem": problem,
            "model": model,
            "model_label": ARTICLE_LABELS.get(model, model),
            "alpha_median_curve": float(alpha_med),
            "log_intercept": float(intercept),
            "alpha_seed_mean": float(np.mean(seed_alphas)) if seed_alphas else float("nan"),
            "alpha_seed_std": float(np.std(seed_alphas, ddof=1)) if len(seed_alphas) > 1 else float("nan"),
            "n_seed_alphas": len(seed_alphas),
        })
    return pd.DataFrame(rows)


def _safe_wilcoxon(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.allclose(a, b):
        return float("nan"), float("nan")
    try:
        stat, pval = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(pval)
    except ValueError:
        return float("nan"), float("nan")


def article_stat_tests(df):
    """Gera os testes descritos no manuscrito.

    - curve_tests: Wilcoxon sobre o eixo dos N usando medianas por N.
    - seed_tests: Wilcoxon pareado por semente em cada N, mais Friedman.
    """
    curve_rows = []
    seed_rows = []

    pairs = [
        ("peds", "peds_afim"),
        ("peds", "nn_only"),
        ("peds_afim", "nn_only"),
    ]

    global_med = df.groupby(["problem", "N", "model"])["test_fe"].median().unstack()
    for a, b in pairs:
        if a in global_med and b in global_med:
            common = global_med[[a, b]].dropna()
            stat, pval = _safe_wilcoxon(common[a], common[b])
            curve_rows.append({
                "scope": "global_problem_N",
                "problem": "all",
                "N": "all",
                "contrast": f"{ARTICLE_LABELS[a]} vs {ARTICLE_LABELS[b]}",
                "statistic": stat,
                "p_value": pval,
                "median_delta_a_minus_b": float((common[a] - common[b]).median()),
                "n_pairs": int(len(common)),
            })
    if all(m in global_med for m in ARTICLE_MODELS):
        common = global_med[list(ARTICLE_MODELS)].dropna()
        if len(common) >= 2:
            try:
                stat, pval = stats.friedmanchisquare(
                    common["peds"], common["peds_afim"], common["nn_only"]
                )
            except ValueError:
                stat, pval = float("nan"), float("nan")
            curve_rows.append({
                "scope": "global_problem_N",
                "problem": "all",
                "N": "all",
                "contrast": "PEDS físico vs PEDS-afim vs NN-only",
                "statistic": float(stat),
                "p_value": float(pval),
                "median_delta_a_minus_b": float("nan"),
                "n_pairs": int(len(common)),
            })

    for problem, sub in df.groupby("problem"):
        med = sub.groupby(["N", "model"])["test_fe"].median().unstack()
        for a, b in pairs:
            if a in med and b in med:
                common = med[[a, b]].dropna()
                stat, pval = _safe_wilcoxon(common[a], common[b])
                curve_rows.append({
                    "scope": "curve_over_N",
                    "problem": problem,
                    "N": "all",
                    "contrast": f"{ARTICLE_LABELS[a]} vs {ARTICLE_LABELS[b]}",
                    "statistic": stat,
                    "p_value": pval,
                    "median_delta_a_minus_b": float((common[a] - common[b]).median()),
                    "n_pairs": int(len(common)),
                })

        for N, sn in sub.groupby("N"):
            pivot = sn.pivot_table(index="seed", columns="model", values="test_fe", aggfunc="mean")
            for a, b in pairs:
                if a in pivot and b in pivot:
                    common = pivot[[a, b]].dropna()
                    stat, pval = _safe_wilcoxon(common[a], common[b])
                    seed_rows.append({
                        "scope": "paired_seeds_at_N",
                        "problem": problem,
                        "N": int(N),
                        "contrast": f"{ARTICLE_LABELS[a]} vs {ARTICLE_LABELS[b]}",
                        "statistic": stat,
                        "p_value": pval,
                        "median_delta_a_minus_b": float((common[a] - common[b]).median()),
                        "n_pairs": int(len(common)),
                    })
            if all(m in pivot for m in ARTICLE_MODELS):
                common = pivot[list(ARTICLE_MODELS)].dropna()
                if len(common) >= 2:
                    try:
                        stat, pval = stats.friedmanchisquare(
                            common["peds"], common["peds_afim"], common["nn_only"]
                        )
                    except ValueError:
                        stat, pval = float("nan"), float("nan")
                    seed_rows.append({
                        "scope": "friedman_seeds_at_N",
                        "problem": problem,
                        "N": int(N),
                        "contrast": "PEDS físico vs PEDS-afim vs NN-only",
                        "statistic": float(stat),
                        "p_value": float(pval),
                        "median_delta_a_minus_b": float("nan"),
                        "n_pairs": int(len(common)),
                    })

    return pd.DataFrame(curve_rows), pd.DataFrame(seed_rows)


def save_article_analysis_tables(df, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    agg = aggregate_article_results(df)
    alphas = fit_learning_exponents(df)
    curve_tests, seed_tests = article_stat_tests(df)

    paths = {}
    for name, table in [
        ("article_summary.csv", agg),
        ("article_exponents.csv", alphas),
        ("article_curve_wilcoxon.csv", curve_tests),
        ("article_seed_tests.csv", seed_tests),
    ]:
        path = os.path.join(save_dir, name)
        table.to_csv(path, index=False)
        paths[name] = path
    return agg, alphas, curve_tests, seed_tests, paths


def plot_article_learning_curves(df, save_dir="./figs"):
    os.makedirs(save_dir, exist_ok=True)
    problems = [p for p in EXPERIMENTS if p in set(df["problem"])]
    ncols = 2
    nrows = int(np.ceil(len(problems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * nrows), squeeze=False)

    agg = aggregate_article_results(df)
    for ax, problem in zip(axes.ravel(), problems):
        sub = agg[agg["problem"] == problem]
        for model in ARTICLE_MODELS:
            sm = sub[sub["model"] == model].sort_values("N")
            if sm.empty:
                continue
            x = sm["N"].to_numpy(dtype=float)
            y = sm["median_fe"].to_numpy(dtype=float)
            lo = sm["q25_fe"].to_numpy(dtype=float)
            hi = sm["q75_fe"].to_numpy(dtype=float)
            ax.loglog(x, y, marker="o", color=ARTICLE_COLORS[model], label=ARTICLE_LABELS[model])
            ax.fill_between(x, lo, hi, color=ARTICLE_COLORS[model], alpha=0.14)
        ax.axhline(0.05, color="gray", ls=":", lw=1, label="erro-alvo 5%")
        ax.set_title(problem)
        ax.set_xlabel("N treino")
        ax.set_ylabel("FE teste")
        ax.grid(alpha=0.3, which="both")

    for ax in axes.ravel()[len(problems):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("PEDS físico vs PEDS-afim vs NN-only", y=1.02, fontweight="bold")
    path = os.path.join(save_dir, "article_learning_curves.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.show()
    return path


def _plot_table(df, title, path, columns=None, max_rows=18):
    if columns is not None:
        df = df[columns]
    shown = df.head(max_rows).copy()
    fig, ax = plt.subplots(figsize=(11, 0.7 + 0.42 * (len(shown) + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=shown.values,
        colLabels=shown.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)
    for j in range(len(shown.columns)):
        tbl[0, j].set_facecolor("#333")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title(title, pad=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.show()
    return path


def plot_article_result_tables(alphas, curve_tests, save_dir="./figs"):
    os.makedirs(save_dir, exist_ok=True)
    alpha_show = alphas.copy()
    for col in ["alpha_median_curve", "alpha_seed_mean", "alpha_seed_std"]:
        alpha_show[col] = alpha_show[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    alpha_path = _plot_table(
        alpha_show,
        "Expoentes de escala FE ~ N^alpha",
        os.path.join(save_dir, "article_alpha_table.png"),
        columns=["problem", "model_label", "alpha_median_curve", "alpha_seed_mean", "alpha_seed_std"],
    )

    tests_show = curve_tests.copy()
    for col in ["statistic", "p_value", "median_delta_a_minus_b"]:
        tests_show[col] = tests_show[col].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
    tests_path = _plot_table(
        tests_show,
        "Wilcoxon sobre a curva de aprendizado (medianas por N)",
        os.path.join(save_dir, "article_curve_wilcoxon_table.png"),
        columns=["problem", "contrast", "p_value", "median_delta_a_minus_b", "n_pairs"],
    )
    return [alpha_path, tests_path]


def run_article_pipeline(data_root, device="cpu", save_root="notebooks",
                         sizes=(64, 128, 256, 512, 1024, 2048, 4096),
                         seeds=range(10), max_epochs=400, lr=5e-5,
                         batch=64, eval_every=10, patience=8,
                         min_epochs=40, verbose=True):
    """Atalho para gerar os artefatos das seções 4 e 5 do manuscrito."""
    figs = os.path.join(save_root, "figs")
    results_dir = os.path.join(save_root, "results")
    df = run_article_sweep(
        data_root, sizes=sizes, seeds=seeds, device=device,
        max_epochs=max_epochs, lr=lr, batch=batch,
        eval_every=eval_every, patience=patience,
        min_epochs=min_epochs, save_dir=results_dir,
        verbose=verbose,
    )
    agg, alphas, curve_tests, seed_tests, paths = save_article_analysis_tables(df, results_dir)
    curve_path = plot_article_learning_curves(df, figs)
    table_paths = plot_article_result_tables(alphas, curve_tests, figs)
    paths["article_learning_curves.png"] = curve_path
    paths["article_tables"] = table_paths
    return {
        "raw": df,
        "summary": agg,
        "alphas": alphas,
        "curve_tests": curve_tests,
        "seed_tests": seed_tests,
        "paths": paths,
    }
