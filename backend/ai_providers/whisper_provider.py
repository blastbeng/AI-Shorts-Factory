import os
import yaml
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger
import librosa

class WhisperProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("speech", {}).get("whisper", {})
        self.gm = GPUManager()
        self.model = None
        self.processor = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, audio_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Whisper non installato.")
            
        gpu = self.gm.get_gpu_for_task("speech_recognition")
        if not gpu:
            raise RuntimeError("Nessuna GPU assegnata per lo speech recognition.")
            
        device = self.gm.get_device_string(gpu['id'], preferred_backend=self.model_info.get("backend"))
        
        if self.model is None:
            logger.info("Caricamento modello Whisper...")
            model_path = self.model_info.get("path")
            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
            
        logger.info(f"Trascrizione audio da: {audio_path}")
        audio, sampling_rate = librosa.load(audio_path, sr=16000)
        
        inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt").to(device)
        
        with torch.no_grad():
            predicted_ids = self.model.generate(**inputs)
            
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.info(f"Trascrizione completata: {transcription}")
        return transcription

    def get_capabilities(self):
        return {"type": "speech", "model": "whisper"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}
