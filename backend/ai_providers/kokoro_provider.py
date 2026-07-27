import os
import yaml
import torch
import scipy.io.wavfile as wavfile
from transformers import AutoTokenizer, AutoModelForCausalLM
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

class KokoroProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("voice", {}).get("kokoro_tts", {})
        self.gm = GPUManager()
        self.model = None
        self.tokenizer = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, text: str, output_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Kokoro TTS non installato.")
            
        gpu = self.gm.get_gpu_for_task("voice_generation")
        if not gpu:
            raise RuntimeError("Nessuna GPU assegnata per la voice generation.")
            
        device = f"cuda:{gpu['id']}" if gpu["backend"] == "cuda" else f"rocm:{gpu['id']}"
        
        if self.model is None:
            logger.info("Caricamento modello Kokoro TTS...")
            model_path = self.model_info.get("path")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            
        logger.info(f"Generazione voce per testo: {text}")
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            audio_output = self.model.generate(**inputs, max_new_tokens=1000)
            
        # Assumendo che il modello restituisca un tensore audio
        # La logica di salvataggio dipende dall'output specifico del modello Kokoro
        audio_data = audio_output.cpu().numpy()
        wavfile.write(output_path, 24000, audio_data)
        logger.info(f"Voce salvata in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "voice", "model": "kokoro_tts"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}
