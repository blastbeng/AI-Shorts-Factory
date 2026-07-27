import yaml
import subprocess

class GPUManager:
    def __init__(self, config_path="configs/gpu.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_gpus(self):
        return self.config.get("gpus", [])

    def get_gpu_for_task(self, task_name):
        for gpu in self.get_gpus():
            if task_name in gpu.get("assigned_tasks", []):
                return gpu
        return None

    def monitor_vram(self, gpu_id):
        gpu = next((g for g in self.get_gpus() if g["id"] == gpu_id), None)
        if not gpu:
            return None
        # TODO: implementare lettura reale con rocm-smi o nvidia-smi
        return {"gpu_id": gpu_id, "vram_total_gb": gpu["vram_gb"], "vram_used_gb": 0, "vram_free_gb": gpu["vram_gb"]}
