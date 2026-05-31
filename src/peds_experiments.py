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
    # split do notebook Julia: valid 1:1024, test end-1023:end, treino 1025:end
    Xv, yv = X[:1024], y[:1024]
    Xtest, ytest = X[-1024:], y[-1024:]
    Xt, yt = X[1024:], y[1024:]
    return (Xt, yt), (Xv, yv), (Xtest, ytest)


# ----------------------- treino com tracking -----------------------
def _train_one(model, Xt, yt, Xtest, ytest, epochs, lr, batch, track_every):
    opt = Adam(model.parameters(), lr=lr)
    n = Xt.shape[0]
    hist_ep, hist_fe = [], []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=Xt.device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = huber(pred, yt[idx])
            loss.backward()
            opt.step()
        if ep % track_every == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                fe = fractional_error(model(Xtest), ytest)
            hist_ep.append(ep + 1); hist_fe.append(fe)
    return hist_ep, hist_fe


def run_diffusion_experiment(name, data_root, device="cpu",
                             epochs=200, lr=5e-5, batch=64, ninit=1088,
                             track_every=5, seed=0):
    """Treina baseline e PEDS, devolve histórico de FE por época e FEs finais."""
    cfg = EXPERIMENTS[name]
    res = cfg["res"]
    sim = DiffusionSim(res)
    (Xt, yt), _, (Xtest, ytest) = load_data(data_root, cfg["data"], device)
    Xt, yt = Xt[:ninit], yt[:ninit]      # regime de poucos dados (~10³)

    torch.manual_seed(seed)
    peds = DiffusionPEDS(res, sim, cw_init=(0.05 if "Fourier" in name else 0.45)).to(device)
    ep_p, fe_p = _train_one(peds, Xt, yt, Xtest, ytest, epochs, lr, batch, track_every)

    torch.manual_seed(seed)
    base = DiffusionBaseline(res).to(device)
    ep_b, fe_b = _train_one(base, Xt, yt, Xtest, ytest, epochs, lr, batch, track_every)

    # FE low-fidelity (coarse puro, w=0) para a Tabela 3
    with torch.no_grad():
        coarse_geom = 1.0 - 0.9 * Xtest
        lowfi = fractional_error(peds.solver(coarse_geom), ytest)

    return {
        "name": name,
        "epochs": ep_p, "fe_peds": fe_p, "fe_nn": fe_b,
        "final_peds": fe_p[-1], "final_nn": fe_b[-1], "lowfi": lowfi,
        "w": float(peds.weight().item()),
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
    """results: dict {nome: {final_peds, final_nn, lowfi, ...}} para os 5 experimentos.
    Reproduz Extended Data Table 2 (PEDS vs NN-only) e Table 3 (PEDS vs low-fidelity)."""
    os.makedirs(save_dir, exist_ok=True)
    order = ["Fourier(16)", "Fourier(25)", "Fisher(16)", "Fisher(25)", "Maxwell(10)"]
    order = [n for n in order if n in results]

    # Tabela 2: PEDS vs NN-only (replicado e paper)
    rows2 = [["Modelo", "PEDS (rep.)", "PEDS (paper)", "NN-only (rep.)", "NN-only (paper)"]]
    for n in order:
        r = results[n]; p = PAPER.get(n, {})
        rows2.append([n, f"{r['final_peds']*100:.1f}%", f"{p.get('peds',float('nan'))*100:.1f}%",
                      f"{r['final_nn']*100:.1f}%", f"{p.get('nn',float('nan'))*100:.1f}%"])

    # Tabela 3: PEDS vs low-fidelity + melhoria
    rows3 = [["Modelo", "PEDS (rep.)", "Low-fi (rep.)", "Low-fi (paper)", "Melhoria (rep.)"]]
    for n in order:
        r = results[n]; p = PAPER.get(n, {})
        imp = r["lowfi"] / r["final_peds"] if r["final_peds"] > 0 else float("nan")
        rows3.append([n, f"{r['final_peds']*100:.1f}%", f"{r['lowfi']*100:.1f}%",
                      f"{p.get('lowfi',float('nan'))*100:.1f}%", f"{imp:.1f}×"])

    paths = []
    for rows, title, fname in [
        (rows2, "Extended Data Table 2 — PEDS vs NN-only (FE no teste)", "tabela2_peds_vs_nn.png"),
        (rows3, "Extended Data Table 3 — PEDS vs low-fidelity", "tabela3_peds_vs_lowfi.png"),
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
