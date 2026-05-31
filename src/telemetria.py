from datetime import datetime
import os
import platform
import time
import pandas as pd
import torch
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetTemperature,
    nvmlInit,
)


class MonitorTreinamento:
    """Gerenciador automático de telemetria e logs de hardware para Deep Learning."""

    def __init__(self, modelo):
        # Captura os dados fixos do sistema na inicialização
        gpu_disponivel = torch.cuda.is_available()
        self.so = f"{platform.system()} {platform.release()}"
        self.gpu_nome = (
            torch.cuda.get_device_name(0) if gpu_disponivel else "N/A"
        )
        self.modelo_name = modelo.__class__.__name__

        # Calcula o total de parâmetros do modelo
        total_params = sum(p.numel() for p in modelo.parameters())
        self.total_parametros = f"{total_params:,}"

        # Inicializa o histórico dinâmico
        self.historico = []
        self.tempo_inicio_epoca = None

    def iniciar_epoca(self):
        """Disparado no começo de cada época para disparar o cronômetro."""
        self.tempo_inicio_epoca = time.time()

    def finalizar_epoca(self, epoca):
        """Captura o tempo decorrido, lê a GPU e armazena na tabela automaticamente."""
        if self.tempo_inicio_epoca is None:
            raise RuntimeError(
                "Você precisa chamar '.iniciar_epoca()' antes de finalizar."
            )

        # 1. Calcula o tempo da época
        duracao = round(time.time() - self.tempo_inicio_epoca, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 2. Captura a telemetria da GPU
        (
            total_vram,
            so_vram,
            ia_vram,
            dispo_vram,
            pct_ia,
            pct_so,
            pct_livre,
            temp,
        ) = self._capturar_gpu()

        # 3. Alimenta a tabela interna
        self.historico.append(
            {
                "Timestamp": timestamp,
                "SO": self.so,
                "GPU_Nome": self.gpu_nome,
                "Modelo_IA": self.modelo_name,
                "Total_Parametros": self.total_parametros,
                "Época": epoca,
                "Tempo_Treino_(s)": duracao,
                "VRAM_Total_(MB)": total_vram,
                "VRAM_SO_(MB)": so_vram,
                "VRAM_IA_(MB)": ia_vram,
                "VRAM_Livre_(MB)": dispo_vram,
                "%_Aloc_IA": f"{pct_ia}%",
                "%_Aloc_SO": f"{pct_so}%",
                "%_Mem_Livre": f"{pct_livre}%",
                "Temp_GPU_(°C)": temp,
            }
        )

    def salvar_logs(self):
        """Consolida os dados em um DataFrame, exibe na tela e exporta para o CSV."""
        df = pd.DataFrame(self.historico)
        os.makedirs("../logs", exist_ok=True)

        timestamp_arq = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_csv = (
            f"../logs/experimento_{self.modelo_name}_{timestamp_arq}.csv"
        )
        df.to_csv(caminho_csv, index=False)

        print(f"\n--- Telemetria salva automaticamente em: {caminho_csv} ---")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1500)
        print(df.to_string(index=False))
        return df

    def _capturar_gpu(self):
        """Método privado interno para ler a API da NVIDIA."""
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)

            vram_total = round(info.total / (1024**2), 2)
            vram_sistema_usada = round(info.used / (1024**2), 2)
            vram_ia_alocada = round(
                torch.cuda.memory_allocated(0) / (1024**2), 2
            )

            vram_so_pura = round(vram_sistema_usada - vram_ia_alocada, 2)
            vram_disponivel_total = round(vram_total - vram_sistema_usada, 2)

            pct_ia = round((vram_ia_alocada / vram_total) * 100, 2)
            pct_so = round((vram_so_pura / vram_total) * 100, 2)
            pct_livre = round(100.0 - (pct_ia + pct_so), 2)

            temperatura = nvmlDeviceGetTemperature(handle, 0)
            return (
                vram_total,
                vram_so_pura,
                vram_ia_alocada,
                vram_disponivel_total,
                pct_ia,
                pct_so,
                pct_livre,
                temperatura,
            )
        except:
            return 0, 0, 0, 0, 0, 0, 0, 0