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
from src.physics.differentiable_physics import DifferentiableMaxwell


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


def run_maxwell_experiment(data_root, device="cpu", epochs=10, lr=1e-3,
                           batch=64, ninit=1280, track_every=1, track_n=256, seed=0):
    phys = _MaxwellPhysics(device)
    (Xt, yt), _, (Xtest, ytest) = _load_maxwell(data_root, device)
    Xt, yt = Xt[:ninit], yt[:ninit]
    Xte, yte = Xtest[:track_n], ytest[:track_n]   # subset p/ FE por época (custo do solver)

    nn_params = NNstruct(
        inGen=[13, 256, 256], outGen=[256, 256, phys.N_BAND * phys.NX],
        postGen=[lambda x: x * 1.5 + 2.5, lambda x: x.reshape(-1, phys.N_BAND, phys.NX)],
        inVar=[phys.N_BAND * phys.NX, 256, 256, 256], outVar=[256, 256, 256, 1],
    )

    def fe_of_peds(model):
        model.eval()
        with torch.no_grad():
            rp, ip, _ = _peds_forward(phys, model, Xte)
            return _fe_complex(rp, ip, yte)

    def fe_of_base(model):
        model.eval()
        with torch.no_grad():
            z = model.mgen(Xte); pred = model.pred(z)
            return _fe_complex(pred[:, 0], pred[:, 1], yte)

    # ---- PEDS ----
    torch.manual_seed(seed)
    peds = PEDSModel(nn_params).to(device)
    opt = Adam(peds.parameters(), lr=lr)
    ep_p, fe_p = [], []
    n = Xt.shape[0]
    for ep in range(epochs):
        peds.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            rp, ip, vp_out = _peds_forward(phys, peds, Xt[idx])
            vp = torch.abs(vp_out.squeeze()) + 1e-6
            loss = calculate_nll_loss((rp, ip, vp), yt[idx])
            loss.backward(); opt.step()
        if ep % track_every == 0 or ep == epochs - 1:
            ep_p.append(ep + 1); fe_p.append(fe_of_peds(peds))

    # ---- baseline NN-only ----
    torch.manual_seed(seed)
    base = BaselineModel(nn_params).to(device)
    optb = Adam(base.parameters(), lr=lr)
    ep_b, fe_b = [], []
    for ep in range(epochs):
        base.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            optb.zero_grad()
            z = base.mgen(Xt[idx]); pred = base.pred(z); vp = base.mvar(z)
            rp, ip = pred[:, 0], pred[:, 1]
            vpp = torch.abs(vp.squeeze()) + 1e-6
            loss = calculate_nll_loss((rp, ip, vpp), yt[idx])
            loss.backward(); optb.step()
        if ep % track_every == 0 or ep == epochs - 1:
            ep_b.append(ep + 1); fe_b.append(fe_of_base(base))

    # FE low-fidelity (coarse puro: w=0)
    peds.eval()
    with torch.no_grad():
        cw_backup = peds.cw.data.clone()
        # força w=0 -> só coarse (downsample), reaproveitando o forward
        band = phys.build_coarse_band(Xte)
        full = phys.buildgeom(band)
        freqs = Xte[:, 10:13] @ torch.tensor([0.5, 0.75, 1.0], device=device)
        rp, ip = phys.solve(full, freqs)
        lowfi = _fe_complex(rp, ip, yte)
        peds.cw.data = cw_backup

    return {
        "name": "Maxwell(10)",
        "epochs": ep_p, "fe_peds": fe_p, "fe_nn": fe_b,
        "final_peds": fe_p[-1], "final_nn": fe_b[-1], "lowfi": lowfi,
        "w": float(torch.sigmoid(peds.cw * peds.multfact).item()),
    }
