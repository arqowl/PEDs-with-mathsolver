#!/usr/bin/env python3
"""Executa o protocolo experimental do artigo PEDS-afim.

O pipeline compara PEDS físico, PEDS-afim e NN-only nos benchmarks de difusão
Fourier/Fisher, gera curvas FE x N, expoentes de escala e testes estatísticos.
"""

import os

import torch

from src.peds_experiments import run_article_pipeline


ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(ROOT, "data")
SAVE_ROOT = os.path.join(ROOT, "notebooks")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
SEEDS = range(10)
MAX_EPOCHS = 400
BATCH = 64
LR = 5e-5
EVAL_EVERY = 10
PATIENCE = 8
MIN_EPOCHS = 40


def main():
    smoke = os.environ.get("PEDS_SMOKE", "").strip() == "1"
    sizes = (64, 128) if smoke else SIZES
    seeds = range(2) if smoke else SEEDS
    max_epochs = 30 if smoke else MAX_EPOCHS
    min_epochs = 10 if smoke else MIN_EPOCHS
    patience = 3 if smoke else PATIENCE

    print(f"device: {DEVICE}")
    print(f"sizes: {tuple(sizes)}")
    print(f"seeds: {list(seeds)}")
    print(f"modo: {'smoke' if smoke else 'completo'}")

    out = run_article_pipeline(
        DATA_ROOT,
        device=DEVICE,
        save_root=SAVE_ROOT,
        sizes=sizes,
        seeds=seeds,
        max_epochs=max_epochs,
        lr=LR,
        batch=BATCH,
        eval_every=EVAL_EVERY,
        patience=patience,
        min_epochs=min_epochs,
        verbose=True,
    )

    print("\nArtefatos gerados:")
    for name, path in out["paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
