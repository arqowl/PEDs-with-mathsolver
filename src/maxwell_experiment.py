# src/maxwell_experiment.py
# Treina PEDS e baseline NN-only para Maxwell(10) registrando o FE de teste
# por época, devolvendo o MESMO formato dos experimentos de difusão para
# entrar nas mesmas curvas de aprendizado e nas mesmas tabelas.
#
# Reaproveita a física já VALIDADA: fdfd.FDFDSolver (operador de Maxwell),
# differentiable_physics.DifferentiableMaxwell (adjunto exato), e a calibração
# do coarse.jl (refsim, frequência por one-hot, embedding buildgeom).

import os
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from src.objects import NNstruct, CSstruct
from src.models.surrogate import PEDSModel, BaselineModel
from src.models.losses import calculate_nll_loss
from src.physics.geometry import SimulationDomain, epsilon_hole_layers
from src.physics.fdfd import FDFDSolver
from src.physics.maxwell_solver import DifferentiableMaxwell   # (era differentiable_physics no original)

try:                                   # telemetria opcional (não quebra sem GPU/pynvml)
    from src.telemetria import MonitorTreinamento
    _HAS_TELE = True
except Exception:
    _HAS_TELE = False


def _fe_complex(rp, ip, y):
    pred = torch.complex(rp, ip)
    return (torch.linalg.norm(pred - y) / torch.linalg.norm(y)).item()


def _load_maxwell(data_root, device):
    Xdf = pd.read_csv(os.path.join(data_root, "X_maxwell10_small.csv"), header=None)
    X = torch.tensor(Xdf.values.T, dtype=torch.float32, device=device)   # (amostras, 13)

    def parse(v):
        return complex(str(v).replace(" ", "").replace("im", "j"))
    yraw = pd.read_csv(os.path.join(data_root, "y_maxwell10_small.csv"), header=None).values.flatten()
    y = torch.tensor(np.array([parse(v) for v in yraw]), dtype=torch.complex64, device=device)

    Xv, yv = X[:1024], y[:1024]
    Xtest, ytest = X[-1024:], y[-1024:]
    Xt, yt = X[1024:], y[1024:]
    return (Xt, yt), (Xv, yv), (Xtest, ytest)


class _MaxwellPhysics:
    """Encapsula grade, solvers por frequência, coarse e buildgeom (calibrado)."""
    def __init__(self, device):
        self.device = device
        self.cs = CSstruct(resolution=10, nn_x=10, ny_nn=110,
                           refsim=complex(0.3364246930443735, 0.1920021246559511))
        cs = self.cs
        sd0 = SimulationDomain(Lx=cs.Lx, Ly=cs.Ly, omega=2 * np.pi, dpml=cs.dpml,
                               resolution=cs.resolution, source=cs.source, monitor=cs.monitor)
        self.NY, self.NX = sd0.ny, sd0.nx
        self.XS, self.YS = sd0.xs, sd0.ys
        self.delta = float(self.XS[1] - self.XS[0])
        self.Lx_grid = float(self.XS[-1] - self.XS[0] + self.delta)
        self.REFSIM = complex(cs.refsim)
        # máscaras magic-index (ar / banda da rede / substrato)
        self._air = (cs.dpml + self.YS) <= 0.35 * (cs.Ly + 2 * cs.dpml)
        self._band = (0.35 * (cs.Ly + 2 * cs.dpml) < (cs.dpml + self.YS)) & \
                     ((cs.dpml + self.YS) <= 0.35 * (cs.Ly + 2 * cs.dpml) + cs.ny_nn / cs.resolution)
        self.N_AIR = int(self._air.sum()); self.N_BAND = int(self._band.sum())
        self.N_SUB = int((~self._air & ~self._band).sum())
        self._maxwell_cache = {}

    def get_maxwell(self, freq):
        k = round(float(freq), 6)
        if k not in self._maxwell_cache:
            cs = self.cs
            sd = SimulationDomain(Lx=cs.Lx, Ly=cs.Ly, omega=2 * np.pi * k, dpml=cs.dpml,
                                  resolution=cs.resolution, source=cs.source, monitor=cs.monitor)
            self._maxwell_cache[k] = DifferentiableMaxwell(FDFDSolver(sd)).to(self.device)
        return self._maxwell_cache[k]

    def build_coarse_band(self, X):
        widths = X[:, :10].detach().cpu().numpy().astype(np.float64)
        widths = np.clip(widths, self.delta + 1e-3, self.Lx_grid - 1e-3)
        bands = []
        for ps in widths:
            g = np.real(epsilon_hole_layers(self.XS, self.YS, ps,
                        refractive_indexes=tuple(self.cs.refracsim),
                        interstice=self.cs.interstice, hole=self.cs.hole))
            bands.append(g[self._band, :])
        return torch.tensor(np.stack(bands, 0), dtype=torch.float32, device=X.device)

    def buildgeom(self, band):
        B = band.shape[0]
        air = band.new_ones(B, self.N_AIR, self.NX)
        sub = band.new_full((B, self.N_SUB, self.NX), self.cs.epssub)
        return torch.cat([air, band, sub], dim=1)

    def solve(self, full, freqs):
        B = full.shape[0]
        out_r = full.new_zeros(B); out_i = full.new_zeros(B)
        for fval in torch.unique(freqs):
            idx = (freqs == fval).nonzero(as_tuple=True)[0]
            r, i = self.get_maxwell(float(fval))(full[idx])
            out_r = out_r.index_copy(0, idx, r); out_i = out_i.index_copy(0, idx, i)
        a, b = self.REFSIM.real, self.REFSIM.imag; d = a * a + b * b
        return (out_r * a + out_i * b) / d, (out_i * a - out_r * b) / d


def _peds_forward(phys, model, X):
    coarse_band = phys.build_coarse_band(X)
    eps_band, vp = model(X, coarse_band)
    full = phys.buildgeom(eps_band)
    freqs = X[:, 10:13] @ torch.tensor([0.5, 0.75, 1.0], device=X.device)
    rp, ip = phys.solve(full, freqs)
    return rp, ip, vp


def _make_nn_params(phys):
    return NNstruct(
        inGen=[13, 256, 256], outGen=[256, 256, phys.N_BAND * phys.NX],
        postGen=[lambda x: x * 1.5 + 2.5, lambda x: x.reshape(-1, phys.N_BAND, phys.NX)],
        inVar=[phys.N_BAND * phys.NX, 256, 256, 256], outVar=[256, 256, 256, 1],
    )


def _predict(phys, model, X, kind):
    """Predição complexa (rp + i·ip) em modo eval."""
    model.eval()
    with torch.no_grad():
        if kind == "peds":
            rp, ip, _ = _peds_forward(phys, model, X)
        else:
            z = model.mgen(X); pred = model.pred(z); rp, ip = pred[:, 0], pred[:, 1]
    return torch.complex(rp, ip)


def _train_maxwell_member(kind, phys, nn_params, Xt, yt, Xte, epochs, lr, batch,
                          grad_clip, seed, track_every=1, tele_name=None):
    """Treina UM modelo (PEDS ou baseline) e devolve o modelo + predições de teste por época."""
    torch.manual_seed(seed)
    model = (PEDSModel(nn_params) if kind == "peds" else BaselineModel(nn_params)).to(Xt.device)
    opt = Adam(model.parameters(), lr=lr)
    mon = MonitorTreinamento(model) if (tele_name and _HAS_TELE) else None
    if mon: mon.modelo_name = tele_name
    n = Xt.shape[0]; ep_list = []; preds = []
    for ep in range(epochs):
        if mon: mon.iniciar_epoca()
        model.train()
        perm = torch.randperm(n, device=Xt.device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            if kind == "peds":
                rp, ip, vp_out = _peds_forward(phys, model, Xt[idx])
                vp = torch.abs(vp_out.squeeze()) + 1e-6
            else:
                z = model.mgen(Xt[idx]); pred = model.pred(z); vpn = model.mvar(z)
                rp, ip = pred[:, 0], pred[:, 1]; vp = torch.abs(vpn.squeeze()) + 1e-6
            loss = calculate_nll_loss((rp, ip, vp), yt[idx])
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        if mon: mon.finalizar_epoca(ep + 1)
        if ep % track_every == 0 or ep == epochs - 1:
            ep_list.append(ep + 1); preds.append(_predict(phys, model, Xte, kind))
    if mon: mon.salvar_logs()
    return model, ep_list, preds


def _ensemble_fe_curve(kind, phys, nn_params, Xt, yt, Xte, yte, n_ensemble,
                       epochs, lr, batch, grad_clip, seed, track_every, telemetry):
    """Treina n_ensemble membros e combina por época: predição = média dos membros."""
    members_preds, ep_ref, ws = [], None, []
    for k in range(n_ensemble):
        tag = f"Maxwell10_{'PEDS' if kind == 'peds' else 'NNonly'}_m{k}" if telemetry else None
        model, ep, preds = _train_maxwell_member(kind, phys, nn_params, Xt, yt, Xte,
                                                 epochs, lr, batch, grad_clip,
                                                 seed + k, track_every, tag)
        ep_ref = ep; members_preds.append(preds)
        if kind == "peds":
            ws.append(float(torch.sigmoid(model.cw * model.multfact).item()))
    fe = []
    for j in range(len(ep_ref)):
        mean_pred = torch.stack([members_preds[k][j] for k in range(n_ensemble)]).mean(0)
        fe.append((torch.linalg.norm(mean_pred - yte) / torch.linalg.norm(yte)).item())
    w_mean = (sum(ws) / len(ws)) if ws else float("nan")
    return ep_ref, fe, w_mean


def run_maxwell_experiment(data_root, device="cpu", epochs=10, lr=3e-4, batch=64,
                           ninit=1280, track_every=1, track_n=256, seed=0,
                           grad_clip=1.0, telemetry=False, n_ensemble=1):
    """Treino estático do Maxwell. n_ensemble>1 = média de n_ensemble modelos (como no paper)."""
    phys = _MaxwellPhysics(device)
    (Xt, yt), _, (Xtest, ytest) = _load_maxwell(data_root, device)
    Xt, yt = Xt[:ninit], yt[:ninit]
    Xte, yte = Xtest[:track_n], ytest[:track_n]
    nn_params = _make_nn_params(phys)

    ep_p, fe_p, w_val = _ensemble_fe_curve("peds", phys, nn_params, Xt, yt, Xte, yte,
                                           n_ensemble, epochs, lr, batch, grad_clip,
                                           seed, track_every, telemetry)
    ep_b, fe_b, _ = _ensemble_fe_curve("base", phys, nn_params, Xt, yt, Xte, yte,
                                       n_ensemble, epochs, lr, batch, grad_clip,
                                       seed, track_every, telemetry)
    with torch.no_grad():                                  # low-fi (coarse puro)
        band = phys.build_coarse_band(Xte); full = phys.buildgeom(band)
        freqs = Xte[:, 10:13] @ torch.tensor([0.5, 0.75, 1.0], device=device)
        rp, ip = phys.solve(full, freqs); lowfi = _fe_complex(rp, ip, yte)

    return {"name": "Maxwell(10)", "epochs": ep_p, "fe_peds": fe_p, "fe_nn": fe_b,
            "final_peds": fe_p[-1], "final_nn": fe_b[-1], "lowfi": lowfi,
            "w": w_val, "n_ensemble": n_ensemble}


# ============================ ACTIVE LEARNING (baseado em pool) ============================
def run_maxwell_active_learning(data_root, device="cpu", n_ensemble=3, ninit=256,
                                T=6, M=4, K=128, epochs=8, lr=3e-4, batch=64,
                                grad_clip=1.0, track_n=256, seed=0,
                                compare_random=True, compare_nn=True):
    """Active learning do PEDS (Maxwell), baseado em POOL.

    A cada iteração, treina um ensemble e adiciona ao treino os K pontos de MAIOR
    incerteza (discordância entre os membros do ensemble) escolhidos de um pool de
    M*K candidatos — reproduzindo a estratégia de aquisição do paper (ninit/T/M/K),
    porém selecionando de pontos JÁ rotulados (sem o solver de alta-fidelidade para
    rotular pontos novos). Compara contra aquisição ALEATÓRIA no mesmo orçamento.

    ATENÇÃO: é o experimento mais pesado (ensemble × T × solver FDFD). Use defaults
    modestos e aumente conforme a sua máquina aguentar.
    """
    phys = _MaxwellPhysics(device)
    (Xpool, ypool), _, (Xtest, ytest) = _load_maxwell(data_root, device)
    Xte, yte = Xtest[:track_n], ytest[:track_n]
    nn_params = _make_nn_params(phys)
    pool_n = Xpool.shape[0]

    def run(kind, acq):
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(pool_n, generator=g).tolist()
        in_train, remaining = perm[:ninit], perm[ninit:]
        pts, fes = [], []
        for t in range(T + 1):
            idx = torch.tensor(in_train, device=device)
            Xt, yt = Xpool[idx], ypool[idx]
            members, preds_test = [], []
            for k in range(n_ensemble):
                m, _, pr = _train_maxwell_member(kind, phys, nn_params, Xt, yt, Xte,
                                                 epochs, lr, batch, grad_clip,
                                                 seed + 100 * t + k, track_every=epochs)
                members.append(m); preds_test.append(pr[-1])
            mean_test = torch.stack(preds_test).mean(0)
            fe = (torch.linalg.norm(mean_test - yte) / torch.linalg.norm(yte)).item()
            pts.append(len(in_train)); fes.append(fe)
            print(f"    [{kind}/{acq}] iter {t}: N={len(in_train)} FE={fe:.3f}")
            if t == T or not remaining:
                break
            cand = remaining[:M * K]
            if acq == "al":                          # incerteza = discordância do ensemble
                ct = torch.tensor(cand, device=device)
                pm = torch.stack([_predict(phys, m, Xpool[ct], kind) for m in members])
                disagree = pm.real.std(0) + pm.imag.std(0)
                topk = torch.topk(disagree, min(K, len(cand))).indices.tolist()
                chosen = [cand[i] for i in topk]
            else:                                    # aleatório
                chosen = cand[:K]
            chosen_set = set(chosen)
            in_train = in_train + chosen
            remaining = [r for r in remaining if r not in chosen_set]
        return pts, fes

    out = {"name": "Maxwell(10) — Active Learning"}
    print("  >>> PEDS + AL (aquisição por incerteza)")
    out["points"], out["fe_al"] = run("peds", "al")
    if compare_random:
        print("  >>> PEDS (aquisição aleatória = estático)")
        _, out["fe_peds"] = run("peds", "random")
    if compare_nn:
        print("  >>> NN-only (aquisição aleatória)")
        _, out["fe_nn"] = run("base", "random")
    return out


def plot_al_curve(result, target=0.20, save_dir="./figs"):
    import os
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if "fe_nn" in result:
        ax.loglog(result["points"], result["fe_nn"], "--", color="#1f77b4",
                  marker="x", label="NN-only")
    if "fe_peds" in result:
        ax.loglog(result["points"], result["fe_peds"], "-", color="#2ca02c",
                  marker="s", ms=4, label="PEDS")
    ax.loglog(result["points"], result["fe_al"], "-", color="#d62728",
              marker="o", ms=4, label="PEDS + AL")
    ax.axhline(target, color="gray", ls=":", label=f"alvo {target*100:.0f}%")
    ax.set_xlabel("nº de pontos de treino"); ax.set_ylabel("Fractional Error (teste)")
    ax.set_title("Maxwell(10) — eficiência de dados (NN-only vs PEDS vs PEDS+AL)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    path = os.path.join(save_dir, "active_learning_Maxwell10.png")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.show()
    return path
