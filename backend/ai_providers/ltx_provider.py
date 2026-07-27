import os
import yaml
import torch
import numpy as np
from diffusers import LTXVideoPipeline
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger
import imageio

class LtxProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("video", {}).get("ltx_video", {})
        self.gm = GPUManager()
        self.pipeline = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompt: str, output_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello LTX Video non installato.")
            
        gpu = self.gm.get_gpu_for_task("video_generation", self.get_gpu_requirements().get("vram_required_gb", 0))
        if not gpu:
            raise RuntimeError("Nessuna GPU assegnata per la video generation.")
            
        device = self.gm.get_device_string(gpu['id'], preferred_backend=self.model_info.get("backend"))
        
        if self.pipeline is None:
            logger.info("Caricamento pipeline LTX Video...")
            model_path = os.path.join(self.model_info.get("path"), "ltx-video-2b-v0.9.5")
            self.pipeline = LTXVideoPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
            self.pipeline.to(device)
            
        logger.info(f"Generazione video LTX per prompt: {prompt}")
        video = self.pipeline(
            prompt, 
            num_inference_steps=30, 
            height=1920, 
            width=1080,
            num_frames=49
        ).frames[0]

        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()

        imageio.mimsave(output_path, video, fps=24)
        logger.info(f"Video LTX salvato in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "video", "model": "ltx_video"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}
