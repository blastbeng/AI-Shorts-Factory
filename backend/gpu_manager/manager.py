import os
import yaml
import subprocess
import json
from backend.services.logger import logger

class GPUManager:
    def __init__(self, config_path=None):
        path = config_path or os.getenv("GPU_CONFIG_PATH", "configs/gpu.yaml")
        with open(path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_gpus(self):
        gpus = self.config.get("gpus", [])
        # Ordina per VRAM decrescente per prioritizzare la GPU più potente
        return sorted(gpus, key=lambda g: g.get("vram_gb", 0), reverse=True)

    def get_gpu_for_task(self, task_name, required_vram_gb=0):
        if required_vram_gb is None:
            required_vram_gb = 0
            
        logger.info(f"Ricerca GPU per task '{task_name}' con requisito VRAM: {required_vram_gb}GB")
        
        import time
        max_retries = 3
        retry_delay = 3  # seconds
        
        for attempt in range(max_retries):
            for gpu in self.get_gpus():
                if task_name in gpu.get("assigned_tasks", []):
                    vram_info = self.monitor_vram(gpu["id"])
                    if vram_info:
                        logger.info(f"GPU {gpu['id']} ({gpu['name']}) ha {vram_info['vram_free_gb']}GB liberi. Richiesti: {required_vram_gb}GB.")
                        if vram_info["vram_free_gb"] >= required_vram_gb:
                            logger.info(f"GPU {gpu['id']} assegnata per '{task_name}'.")
                            return gpu
                        else:
                            logger.warning(f"GPU {gpu['id']} scartata per VRAM insufficiente.")
                    else:
                        logger.warning(f"Impossibile ottenere info VRAM per GPU {gpu['id']}.")
                else:
                    logger.debug(f"GPU {gpu['id']} non assegnata al task '{task_name}'.")
            
            if attempt < max_retries - 1:
                logger.warning(f"Tentativo {attempt + 1}/{max_retries}: Nessuna GPU disponibile con VRAM sufficiente. Attesa di {retry_delay} secondi per il rilascio della VRAM...")
                time.sleep(retry_delay)
                
        logger.error(f"Nessuna GPU disponibile per il task '{task_name}' con {required_vram_gb}GB richiesti dopo {max_retries} tentativi.")
        return None

    def get_gpu_for_task_ignore_vram(self, task_name):
        for gpu in self.get_gpus():
            if task_name in gpu.get("assigned_tasks", []):
                return gpu
        return None

    def get_device_string(self, gpu_id, preferred_backend=None):
        """Restituisce la stringa del dispositivo PyTorch corretta."""
        gpu = next((g for g in self.get_gpus() if g["id"] == gpu_id), None)
        if not gpu:
            return "cpu"
        
        backends = gpu.get("backends", [])
        device_index = gpu.get("device_index", 0)
        
        # Se un backend preferito è specificato e supportato, usalo
        if preferred_backend and preferred_backend in backends:
            if preferred_backend in ["cuda", "rocm"]:
                return f"cuda:{device_index}"
            elif preferred_backend == "vulkan":
                logger.debug("Backend Vulkan selezionato per PyTorch. Uso fallback su CPU.")
                return "cpu"
        
        # Altrimenti, cerca il primo backend supportato da PyTorch (cuda o rocm)
        for b in backends:
            if b in ["cuda", "rocm"]:
                return f"cuda:{device_index}"
        
        # Se nessun backend PyTorch è disponibile, fallback su cpu
        return "cpu"

    def monitor_vram(self, gpu_id):
        gpu = next((g for g in self.get_gpus() if g["id"] == gpu_id), None)
        if not gpu:
            return None

        vram_total = gpu["vram_gb"]
        vram_used = 0
        gpu_util = 0
        backends = gpu.get("backends", [])
        device_index = gpu.get("device_index", 0)

        try:
            if "cuda" in backends:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-i", str(device_index)],
                    capture_output=True, text=True, check=True
                )
                parts = result.stdout.strip().split(", ")
                vram_used_mb = int(parts[0])
                vram_used = vram_used_mb / 1024
                gpu_util = int(parts[1])
            elif "rocm" in backends:
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"],
                    capture_output=True, text=True, check=True
                )
                stdout = result.stdout
                json_start = stdout.find('{')
                json_end = stdout.rfind('}')
                if json_start != -1 and json_end != -1:
                    json_str = stdout[json_start:json_end+1]
                    data = json.loads(json_str)
                else:
                    data = {}
                
                card_data = data.get(f"card{device_index}", {})
                vram_used_bytes = float(card_data.get("VRAM Total Used Memory (B)", 0))
                vram_used = vram_used_bytes / (1024 * 1024 * 1024)
                gpu_util_str = str(card_data.get("GPU use (%)", "0")).replace("%", "").strip()
                gpu_util = int(float(gpu_util_str))
            else:
                logger.debug(f"Nessun backend supportato per il monitoraggio VRAM su GPU {gpu_id}.")
                vram_used = 0
        except Exception as e:
            logger.debug(f"Errore nel monitoraggio VRAM per GPU {gpu_id}: {e}")
            vram_used = 0

        return {
            "gpu_id": gpu_id,
            "vram_total_gb": vram_total,
            "vram_used_gb": round(vram_used, 2),
            "vram_free_gb": round(vram_total - vram_used, 2),
            "gpu_utilization": gpu_util
        }
