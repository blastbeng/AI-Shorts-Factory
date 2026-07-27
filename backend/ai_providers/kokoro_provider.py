import os
import yaml
import torch
import scipy.io.wavfile as wavfile
from kokoro import KModel, KPipeline
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
        self.pipeline = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, text: str, output_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Kokoro TTS non installato.")
            
        reqs = self.get_gpu_requirements()
        required_vram = reqs.get("vram_required_gb", 0)
        logger.info(f"Kokoro richiede {required_vram}GB di VRAM per la voice generation.")
        
        gpu = self.gm.get_gpu_for_task("voice_generation", required_vram)
        if not gpu:
            logger.warning("Nessuna GPU con VRAM sufficiente per Kokoro. Uso GPU con offload su RAM.")
            gpu = self.gm.get_gpu_for_task_ignore_vram("voice_generation")
            if not gpu:
                raise RuntimeError("Nessuna GPU assegnata per la voice generation.")
            use_cpu_offload = True
        else:
            use_cpu_offload = False
            
        device = self.gm.get_device_string(gpu['id'], preferred_backend=self.model_info.get("backend"))
        
        # Mappa la lingua dell'app al codice lingua di Kokoro
        lang_map = {
            "english": "a",
            "italian": "i",
            "spanish": "e",
            "french": "f",
            # "german": "d", # Kokoro non supporta il tedesco nativamente
        }
        app_language = kwargs.get("language", "english").lower()
        kokoro_lang = lang_map.get(app_language, "a") # Fallback a inglese
        
        if app_language not in lang_map:
            logger.warning(f"Lingua {app_language} non supportata nativamente da Kokoro. Uso fallback su inglese ('a').")
            
        if self.model is None:
            logger.info("Caricamento modello Kokoro TTS...")
            try:
                self.model = KModel().to(device)
                self.pipeline = KPipeline(model=self.model, lang_code=kokoro_lang)
            except Exception as e:
                logger.exception(f"Errore nel caricamento del modello Kokoro.")
                raise
                
        import re
        # Pulizia del testo da markdown, asterischi e processi di pensiero residui
        text = re.sub(r'\<think\>.*?\<\/think\>', '', text, flags=re.DOTALL).strip()
        text = re.sub(r'^.*?(Here\'s a thinking process:|Thinking Process:).*?\n', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\*+', '', text).strip()
        text = re.sub(r'#+', '', text).strip()
        
        logger.info(f"Generazione voce per testo: {text}")
        
        # Mappa la lingua al miglior voce Kokoro disponibile
        voice_map = {
            "english": "af_heart",
            "italian": "im_nicola",  # Voce maschile italiana
            "spanish": "ef_dora",    # Voce femminile spagnola
            "french": "ff_siwis"     # Voce femminile francese
        }
        voice_name = kwargs.get("voice_name", voice_map.get(app_language, "af_heart"))
        
        import numpy as np
        
        with torch.no_grad():
            audio_chunks = []
            for graphemes, phonemes, audio in self.pipeline(text, voice=voice_name, speed=1.0):
                audio_chunks.append(audio.cpu().numpy().squeeze())
                
        if not audio_chunks:
            raise RuntimeError("Kokoro non ha generato alcun audio.")
            
        audio_data = np.concatenate(audio_chunks)
        # Converti da float32 [-1, 1] a int16
        audio_data = (audio_data * 32767).astype("int16")
        wavfile.write(output_path, 24000, audio_data)
        logger.info(f"Voce salvata in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "voice", "model": "kokoro_tts"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
