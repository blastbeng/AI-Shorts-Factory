import os
import yaml
import torch
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, AutoModel
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

class MMAudioProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("audio", {}).get("mmaudio", {})
        self.gm = GPUManager()
        self.model = None
        self.processor = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompt: str, output_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello MMAudio non installato.")
            
        gpu = self.gm.get_gpu_for_task("audio_generation")
        if not gpu:
            raise RuntimeError("Nessuna GPU assegnata per l'audio generation.")
            
        device = f"cuda:{gpu['id']}" if gpu["backend"] == "cuda" else f"rocm:{gpu['id']}"
        
        if self.model is None:
            logger.info("Caricamento modello MMAudio...")
            model_path = self.model_info.get("path")
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(device)
            
        logger.info(f"Generazione audio per prompt: {prompt}")
        inputs = self.processor(text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            audio_values = self.model.generate(**inputs)
            
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        wavfile.write(output_path, sampling_rate, audio_values.cpu().numpy())
        logger.info(f"Audio salvato in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "audio", "model": "mmaudio"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}
