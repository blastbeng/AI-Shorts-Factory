import os
import yaml
import torch
from transformers import pipeline
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

class LLMProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        # Assumiamo ci sia una sezione 'text' in models.yaml per l'LLM
        self.model_info = self.models_config.get("text", {}).get("llm_base", {})
        self.gm = GPUManager()
        self.generator = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompt: str, max_length: int = 500, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello LLM non installato.")
            
        gpu = self.gm.get_gpu_for_task("text_generation")
        if not gpu:
            # Fallback su CPU se nessuna GPU assegnata
            device = -1
        else:
            device = gpu['id'] if gpu["backend"] == "cuda" else -1 # Semplificato per CPU/CUDA
            
        if self.generator is None:
            logger.info("Caricamento pipeline LLM...")
            model_path = self.model_info.get("path", "gpt2")
            self.generator = pipeline("text-generation", model=model_path, device=device)
            
        logger.info(f"Generazione testo per prompt: {prompt}")
        result = self.generator(prompt, max_length=max_length, num_return_sequences=1, truncation=True)
        generated_text = result[0]["generated_text"]
        return generated_text

    def get_capabilities(self):
        return {"type": "text", "model": "llm_base"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb", 4), "backend": self.model_info.get("backend", "cuda")}
