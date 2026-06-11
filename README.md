# PEDS-afim

Este repositório implementa o protocolo experimental do artigo **Avaliação Estatística da Eficiência de Dados do PEDS: Uma Abordagem Estrutural com um Surrogate de Baixa Fidelidade Não-Físico**.

O objetivo é testar se a eficiência de dados do PEDS depende da física embutida no solver low-fidelity ou se pode ser explicada pelo acoplamento estrutural entre gerador neural, geometria coarse e gargalo de baixa fidelidade.

## Modelos Comparados

- **PEDS físico**: gerador neural + mistura com geometria coarse + solver de difusão diferenciável.
- **PEDS-afim**: mesmo gerador e mesma mistura coarse, mas substitui o solver físico por uma leitura afim `kappa = a^T geom + b`.
- **NN-only**: baseline puramente data-driven do artigo original.

## Benchmarks

O protocolo usa os benchmarks de difusão do PEDS original:

- `Fourier(16)`
- `Fourier(25)`
- `Fisher(16)`
- `Fisher(25)`

Cada base tem 10.000 amostras. O split fixo é:

- validação: primeiras 1.024 amostras;
- teste: últimas 1.024 amostras;
- treino: bloco intermediário, sem vazamento para o teste.

## Protocolo

O experimento principal varre:

```text
N = 64, 128, 256, 512, 1024, 2048, 4096
seeds = 0..9
batch = 64
optimizer = Adam
loss = Huber
metric = Fractional Error
```

O treino usa early stopping por validação. O pipeline gera:

- resultados granulares por problema, modelo, `N` e seed;
- curvas `FE x N`;
- expoentes de escala `FE ~ N^alpha`;
- testes de Wilcoxon e Friedman;
- figuras e tabelas para as seções 4 e 5 do artigo.

## Como Rodar

Pelo notebook:

```text
notebooks/run_all_experiments.ipynb
```

Pelo terminal:

```bash
python run_all_experiments.py
```

Para um teste rápido de sanidade:

```bash
PEDS_SMOKE=1 python run_all_experiments.py
```

Os artefatos são salvos em:

- `notebooks/results/`
- `notebooks/figs/`

## Estrutura

- `src/peds_experiments.py`: modelos Fourier/Fisher, PEDS-afim, treino, sweeps, estatística e plots.
- `src/physics/diffusion_solver.py`: solver coarse de difusão diferenciável.
- `notebooks/run_all_experiments.ipynb`: execução principal do protocolo do artigo.
- `zfiles/PEDs-afim.pdf`: manuscrito da proposta.
- `zfiles/Physics-enhanced deep surrogates for PDEs.pdf`: artigo PEDS original.

## Telemetria

O módulo `src/telemetria.py` ainda está disponível para monitorar CPU, RAM, GPU e tempo por época, mas a telemetria não faz parte do protocolo estatístico principal do artigo.
