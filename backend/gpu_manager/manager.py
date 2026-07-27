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
        return self.config.get("gpus", [])

    def get_gpu_for_task(self, task_name, required_vram_gb=0):
        for gpu in self.get_gpus():
            if task_name in gpu.get("assigned_tasks", []):
                vram_info = self.monitor_vram(gpu["id"])
                if vram_info and vram_info["vram_free_gb"] >= required_vram_gb:
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
