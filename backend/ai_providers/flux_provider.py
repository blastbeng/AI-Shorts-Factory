import yaml
import torch
from diffusers import FluxPipeline
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

class FluxProvider(BaseAIProvider):
    def __init__(self):
        with open("configs/models.yaml", "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("image", {}).get("flux", {})
        self.gm = GPUManager()
        self.pipeline = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompt: str, output_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Flux non installato.")
            
        gpu = self.gm.get_gpu_for_task("image_generation")
        if not gpu:
            raise RuntimeError("Nessuna GPU assegnata per l'image generation.")
            
        device = f"cuda:{gpu['id']}" if gpu["backend"] == "cuda" else f"rocm:{gpu['id']}"
        
        if self.pipeline is None:
            logger.info("Caricamento pipeline Flux...")
            model_path = self.model_info.get("path")
            self.pipeline = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
            self.pipeline.to(device)
            
        logger.info(f"Generazione immagine per prompt: {prompt}")
        image = self.pipeline(prompt, num_inference_steps=30).images[0]
        
        image.save(output_path)
        logger.info(f"Immagine salvata in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "image", "model": "flux"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}
