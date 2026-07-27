import yaml
import subprocess
import json

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

        vram_total = gpu["vram_gb"]
        vram_used = 0

        try:
            if gpu["backend"] == "cuda":
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu_id)],
                    capture_output=True, text=True, check=True
                )
                vram_used_mb = int(result.stdout.strip())
                vram_used = vram_used_mb / 1024
            elif gpu["backend"] == "rocm":
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--json"],
                    capture_output=True, text=True, check=True
                )
                data = json.loads(result.stdout)
                vram_used_mb = int(data.get(f"card{gpu_id}", {}).get("VRAM Total Used Memory (B)", 0)) / (1024 * 1024)
                vram_used = vram_used_mb / 1024
        except Exception as e:
            print(f"Errore nel monitoraggio VRAM per GPU {gpu_id}: {e}")
            vram_used = 0

        return {
            "gpu_id": gpu_id,
            "vram_total_gb": vram_total,
            "vram_used_gb": round(vram_used, 2),
            "vram_free_gb": round(vram_total - vram_used, 2)
        }
