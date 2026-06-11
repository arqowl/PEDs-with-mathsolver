from datetime import datetime
import os
import platform
import time
import pandas as pd
import torch
import psutil
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetTemperature,
    nvmlInit,
)


class MonitorTreinamento:
    """Gerenciador automático de telemetria e logs de hardware (CPU, RAM e GPU) para Deep Learning."""

    def __init__(self, modelo):
        # Captura os dados fixos do sistema na inicialização
        gpu_disponivel = torch.cuda.is_available()
        self.so = f"{platform.system()} {platform.release()}"
        self.gpu_nome = (
            torch.cuda.get_device_name(0) if gpu_disponivel else "N/A"
        )
        
        # Captura a quantidade total de núcleos lógicos da CPU
        self.cpu_nucleos = psutil.cpu_count(logical=True)
        self.modelo_name = modelo.__class__.__name__

        # Calcula o total de parâmetros do modelo
        total_params = sum(p.numel() for p in modelo.parameters())
        self.total_parametros = f"{total_params:,}"

        # Inicializa o histórico dinâmico
        self.historico = []
        self.tempo_inicio_epoca = None

    def iniciar_epoca(self):
        """Disparado no começo de cada época para ligar o cronômetro."""
        # Dá um "reset" nos medidores de CPU para iniciar a integração do tempo ocioso
        psutil.cpu_percent(interval=None)
        self.tempo_inicio_epoca = time.time()

    def finalizar_epoca(self, epoca):
        """Captura o tempo decorrido, processa as partições de 100% do hardware e armazena os dados."""
        if self.tempo_inicio_epoca is None:
            raise RuntimeError(
                "Você precisa chamar '.iniciar_epoca()' antes de finalizar."
            )

        # 1. Tempo e Timestamp do Experimento
        duracao = round(time.time() - self.tempo_inicio_epoca, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 2. Bloco CPU: Processamento baseado em tempo de execução vs tempo ocioso
        pct_cpu_uso = psutil.cpu_percent(interval=None)
        pct_cpu_livre = round(100.0 - pct_cpu_uso, 2)
        
        # Rastreamento de núcleos ativos (acima de 5% de estresse individual)
        usos_por_nucleo = psutil.cpu_percent(percpu=True)
        nucleos_ativos = sum(1 for uso in usos_por_nucleo if uso > 5.0)

        # 3. Bloco RAM: Partição de 100% do espaço de armazenamento global
        (
            ram_total,
            ram_so,
            ram_ia,
            ram_livre,
            pct_ram_ia,
            pct_ram_so,
            pct_ram_livre
        ) = self._capturar_ram_detalhada()

        # 4. Bloco VRAM (GPU): Partição de 100% da memória de vídeo e Temperatura
        (
            total_vram,
            so_vram,
            ia_vram,
            dispo_vram,
            pct_vram_ia,
            pct_vram_so,
            pct_vram_livre,
            temp_gpu,
        ) = self._capturar_gpu_detalhada()

        # 5. Consolidação na Tabela do Histórico
        self.historico.append(
            {
                "Timestamp": timestamp,
                "SO": self.so,
                "GPU_Nome": self.gpu_nome,
                "Modelo_IA": self.modelo_name,
                "Total_Parametros": self.total_parametros,
                "Época": epoca,
                "Tempo_Treino_(s)": duracao,
                
                # --- MÉTRICAS DE CPU ---
                "Uso_CPU_Geral": f"{pct_cpu_uso}%",
                "CPU_Livre": f"{pct_cpu_livre}%",  # Fecha 100% com Uso_CPU_Geral
                "CPU_Nucleos_Totais": self.cpu_nucleos,
                "CPU_Nucleos_Ativos": nucleos_ativos,
                
                # --- MÉTRICAS DE RAM (Física) ---
                "RAM_Total_(GB)": ram_total,
                "RAM_SO_(GB)": ram_so,
                "RAM_IA_(GB)": ram_ia,
                "RAM_Livre_(GB)": ram_livre,
                "RAM_P_IA": f"{pct_ram_ia}%",      # \
                "RAM_P_SO": f"{pct_ram_so}%",      #  |- Juntas fecham 100% do espaço
                "RAM_P_Livre": f"{pct_ram_livre}%",# /
                
                # --- MÉTRICAS DE VRAM (GPU) ---
                "VRAM_Total_(MB)": total_vram,
                "VRAM_SO_(MB)": so_vram,
                "VRAM_IA_(MB)": ia_vram,
                "VRAM_Livre_(MB)": dispo_vram,
                "GPU_P_IA": f"{pct_vram_ia}%",     # \
                "GPU_P_SO": f"{pct_vram_so}%",     #  |- Juntas fecham 100% do espaço
                "GPU_P_Livre": f"{pct_vram_livre}%",# /
                "Temp_GPU_(°C)": temp_gpu,
            }
        )

    def salvar_logs(self):
        """Exporta os dados históricos para a pasta ./logs na raiz do projeto e imprime a tabela."""
        df = pd.DataFrame(self.historico)
        os.makedirs("./logs", exist_ok=True)

        timestamp_arq = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_csv = (
            f"./logs/experimento_{self.modelo_name}_{timestamp_arq}.csv"
        )
        df.to_csv(caminho_csv, index=False)

        print(f"\n--- Telemetria Consolidada com Sucesso: {caminho_csv} ---")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 2500)
        print(df.to_string(index=False))
        return df

    def _capturar_ram_detalhada(self):
        """Mapeia matematicamente a divisão da memória RAM do computador."""
        mem = psutil.virtual_memory()
        processo = psutil.Process(os.getpid())
        
        ram_total = round(mem.total / (1024**3), 2)
        ram_ia = round(processo.memory_info().rss / (1024**3), 2)  # Consumo exclusivo da execução do script
        ram_sistema_usada = round(mem.used / (1024**3), 2)
        
        ram_so = round(ram_sistema_usada - ram_ia, 2)
        if ram_so < 0: ram_so = 0.0  # Tratamento de contorno contra pequenos delays de leitura do kernel
            
        ram_livre = round(ram_total - ram_sistema_usada, 2)
        
        # Força as porcentagens a se basearem rigidamente no teto do hardware total (Soma = 100%)
        pct_ram_ia = round((ram_ia / ram_total) * 100, 2)
        pct_ram_so = round((ram_so / ram_total) * 100, 2)
        pct_ram_livre = round(100.0 - (pct_ram_ia + pct_ram_so), 2)
        
        return ram_total, ram_so, ram_ia, ram_livre, pct_ram_ia, pct_ram_so, pct_ram_livre

    def _capturar_gpu_detalhada(self):
        """Acessa a API de baixo nível do Driver NVIDIA para mapear a VRAM."""
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
            if vram_so_pura < 0: vram_so_pura = 0.0
                
            vram_disponivel_total = round(vram_total - vram_sistema_usada, 2)

            # Força as porcentagens a se basearem rigidamente no teto da VRAM total (Soma = 100%)
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
            # Fallback seguro para evitar travamento do script caso o código rode sem GPU ativa
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0