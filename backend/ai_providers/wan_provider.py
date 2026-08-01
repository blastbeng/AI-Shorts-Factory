import os
import gc
import json
import yaml
import torch
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

torch.set_num_threads(1)  # reduce CPU memory overhead from parallel operations


class WanProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("video", {}).get("wan_2_2_14b", {})
        self.offload_strategy = self.model_info.get("offload_strategy", "sequential")
        self.gm = GPUManager()
        self.pipeline = None
        self.base_seed = 42
        # Paths
        self.model_path = os.path.abspath(self.model_info.get("path"))
        self.base_model_path = os.path.abspath(self.model_info.get("base_model_path"))

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompts: list, output_path: str, *args, frames_per_clip=int(os.getenv("GEN_FRAMES", 49)), width=int(os.getenv("GEN_WIDTH", 256)), height=int(os.getenv("GEN_HEIGHT", 448)), steps=int(os.getenv("GEN_WAN_STEPS", 40)), **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Wan 2.2 14B non installato.")

        job_id = kwargs.get("job_id")
        image_path = kwargs.get("image_path")

        # Build config for subprocess
        config = {
            "base_model_path": self.base_model_path,
            "model_path": self.model_path,
            "prompts": prompts,
            "output_path": output_path,
            "image_path": image_path,
            "frames_per_clip": frames_per_clip,
            "width": width,
            "height": height,
            "steps": steps,
            "job_id": job_id,
            "base_seed": self.base_seed,
        }

        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        # Path to CUDA venv python
        cuda_python = os.path.join(os.path.dirname(__file__), '..', '..', 'venv_cuda', 'bin', 'python')
        subprocess_script = os.path.join(os.path.dirname(__file__), 'wan_subprocess.py')

        logger.info("Launching Wan subprocess with CUDA on RTX 3060...")
        proc = subprocess.Popen(
            [cuda_python, subprocess_script, config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Monitor progress
        from backend.services.progress_tracker import ProgressTracker
        tracker = ProgressTracker()
        try:
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("PROGRESS:"):
                    parts = line.split(":")
                    if len(parts) >= 6:
                        jid = parts[1]
                        stage = parts[2]
                        current = parts[3]
                        total = parts[4]
                        msg = ":".join(parts[5:])
                        tracker.update(jid, stage, int(current), int(total), msg)
                elif line.startswith("DONE:"):
                    logger.info(f"Subprocess finished: {line}")
                else:
                    logger.info(f"[WanSubprocess] {line}")
            proc.wait()
        finally:
            os.unlink(config_path)

        if proc.returncode != 0:
            raise RuntimeError(f"Wan subprocess failed with return code {proc.returncode}")

        logger.info(f"Video generato con successo: {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "video", "model": "wan_2_2_14b"}

    def get_gpu_requirements(self):
        return {
            "vram_required_gb": self.model_info.get("vram_required_gb", 8),
            "backend": self.model_info.get("backend")
        }

    def cleanup(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
