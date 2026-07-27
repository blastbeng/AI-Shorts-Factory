import os
import yaml
import torch
from diffusers import DiffusionPipeline
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

class WanProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("video", {}).get("wan_2_2_5b", {})
        self.gm = GPUManager()
        self.pipeline = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompt: str, output_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Wan 2.2 non installato.")

        gpu = self.gm.get_gpu_for_task("video_generation")
        if not gpu:
            raise RuntimeError("Nessuna GPU assegnata per la video generation.")

        device = self.gm.get_device_string(gpu['id'])

        if self.pipeline is None:
            logger.info("Caricamento pipeline Wan 2.2...")
            model_path = self.model_info.get("path")
            self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
            self.pipeline.to(device)

        logger.info(f"Generazione video per prompt: {prompt}")
        video = self.pipeline(prompt, num_inference_steps=50).frames[0]

        import imageio
        imageio.mimsave(output_path, video, fps=24)
        logger.info(f"Video salvato in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "video", "model": "wan_2_2_5b"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}
