# Módulo de Telemetria e Monitoramento de Hardware para Deep Learning

Este módulo provê um **sistema de telemetria automatizado e agnóstico** encapsulado na classe `MonitorTreinamento` (localizada em `src/telemetria.py`). O objetivo principal é monitorar, metrificar e documentar a saúde do hardware (especialmente GPUs NVIDIA via API nativa NVML) e o desempenho computacional de modelos de IA ao longo do treinamento.

Toda vez que um experimento é executado, o módulo gera relatórios dinâmicos no terminal e exporta uma planilha estruturada em formato `.csv` para auditoria posterior, eliminando a necessidade de monitoramento manual via terminais externos.

---

## 📊 Métricas Coletadas

O monitor divide as métricas em duas categorias essenciais:

### 1. Metadados do Ambiente (Estáticos)
* **Sistema Operacional:** Identificação da versão e build do SO hospedeiro (ex: Windows 11, Ubuntu 20.04).
* **Especificação do Hardware:** Nome oficial da GPU detectada pelo ecossistema CUDA (ex: NVIDIA GeForce RTX 4060).
* **Identidade do Modelo:** Nome da classe do modelo Python e contagem total de parâmetros (pesos e vieses) em tempo real.

### 2. Telemetria Temporal (Dinâmicas por Época)
* **Tempo de Execução:** Duração exata do processamento da época em segundos.
* **Distribuição de VRAM (Balanço 100%):**
  * **VRAM IA:** Memória explicitamente alocada pelo PyTorch para o modelo e ativações.
  * **VRAM SO:** Memória consumida pelo Sistema Operacional e processos de background.
  * **VRAM Livre:** Margem de segurança disponível na placa de vídeo.
* **Porcentagens Relativas:** Mapeamento percentual de consumo das três fatias acima em relação à capacidade total da GPU.
* **Temperatura:** Sensor térmico direto do núcleo da GPU em graus Celsius (°C).

---

## 🛠️ Pré-requisitos

O ambiente configurado via **Dev Containers** (`.devcontainer.json`) já traz todas as dependências nativas. Caso precise validar o ambiente localmente, certifique-se de incluir no seu `requirements.txt`:

```text
pandas>=2.0.0
torch>=2.1.0
pynvml>=13.0.1