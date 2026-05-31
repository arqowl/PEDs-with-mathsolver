#!/usr/bin/env python3
# run_all_experiments.py
# Roda os experimentos de difusão (Fourier/Fisher), gera as CURVAS DE
# APRENDIZADO (baseline vs PEDS por época) e as DUAS TABELAS da página 21.
#
# Uso:  python run_all_experiments.py
# (Maxwell vem do notebook; passe seu FE final/curva via --maxwell se quiser
#  incluí-lo nas tabelas — veja MAXWELL_RESULT abaixo.)

import os
import torch
from src.peds_experiments import (
    EXPERIMENTS, run_diffusion_experiment, plot_learning_curve, plot_replication_tables,
)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIGS = "./figs"

# Se você treinou o Maxwell no notebook, preencha aqui para entrar nas tabelas.
# Ex.: MAXWELL_RESULT = {"name":"Maxwell(10)","final_peds":0.28,"final_nn":0.54,
#                        "lowfi":1.24,"epochs":[...],"fe_peds":[...],"fe_nn":[...]}
MAXWELL_RESULT = None


def main():
    results = {}
    for name in EXPERIMENTS:                      # Fourier(16/25), Fisher(16/25)
        print(f"\n=== {name} ===")
        r = run_diffusion_experiment(name, DATA_ROOT, device=DEVICE)
        results[name] = r
        path = plot_learning_curve(r, FIGS)
        print(f"  PEDS FE={r['final_peds']:.3f}  NN-only FE={r['final_nn']:.3f} "
              f"low-fi={r['lowfi']:.3f}  w={r['w']:.3f}")
        print(f"  curva: {path}")

    if MAXWELL_RESULT is not None:
        results["Maxwell(10)"] = MAXWELL_RESULT
        if "epochs" in MAXWELL_RESULT:
            plot_learning_curve(MAXWELL_RESULT, FIGS)

    tabs = plot_replication_tables(results, FIGS)
    print("\nTabelas de replicação:")
    for t in tabs:
        print(" ", t)


if __name__ == "__main__":
    main()
