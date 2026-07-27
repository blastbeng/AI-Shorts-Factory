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
        
        # Se un backend preferito è specificato e supportato, usalo
        if preferred_backend and preferred_backend in backends:
            if preferred_backend in ["cuda", "rocm"]:
                return f"cuda:{gpu['id']}"
            elif preferred_backend == "vulkan":
                logger.debug("Backend Vulkan selezionato per PyTorch. Uso fallback su CPU.")
                return "cpu"
        
        # Altrimenti, cerca il primo backend supportato da PyTorch (cuda o rocm)
        for b in backends:
            if b in ["cuda", "rocm"]:
                return f"cuda:{gpu['id']}"
        
        # Se nessun backend PyTorch è disponibile, fallback su cpu
        return "cpu"

    def monitor_vram(self, gpu_id):
        gpu = next((g for g in self.get_gpus() if g["id"] == gpu_id), None)
        if not gpu:
            return None

        vram_total = gpu["vram_gb"]
        vram_used = 0
        backends = gpu.get("backends", [])
        backend = backends[0] if backends else "cpu"

        try:
            if backend == "cuda":
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu_id)],
                    capture_output=True, text=True, check=True
                )
                vram_used_mb = int(result.stdout.strip())
                vram_used = vram_used_mb / 1024
            elif backend == "rocm":
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--json"],
                    capture_output=True, text=True, check=True
                )
                data = json.loads(result.stdout)
                vram_used_mb = int(data.get(f"card{gpu_id}", {}).get("VRAM Total Used Memory (B)", 0)) / (1024 * 1024)
                vram_used = vram_used_mb / 1024
            elif backend == "vulkan":
                # Non esiste uno strumento standard CLI per monitorare la VRAM Vulkan.
                # Restituiamo un valore fittizio per ora.
                logger.debug("Monitoraggio VRAM per Vulkan non supportato via CLI.")
                vram_used = 0
        except Exception as e:
            logger.error(f"Errore nel monitoraggio VRAM per GPU {gpu_id}: {e}")
            vram_used = 0

        return {
            "gpu_id": gpu_id,
            "vram_total_gb": vram_total,
            "vram_used_gb": round(vram_used, 2),
            "vram_free_gb": round(vram_total - vram_used, 2)
        }
