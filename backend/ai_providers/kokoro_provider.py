import os
import yaml
import torch
import random
import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter
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
        
        preferred_backend = self.model_info.get("backend")
        gpu = self.gm.get_gpu_for_task("voice_generation", required_vram, preferred_backend=preferred_backend)
        if not gpu:
            logger.warning("Nessuna GPU con VRAM sufficiente per Kokoro. Uso GPU con offload su RAM.")
            gpu = self.gm.get_gpu_for_task_ignore_vram("voice_generation", preferred_backend=preferred_backend)
            if not gpu:
                raise RuntimeError("Nessuna GPU assegnata per la voice generation.")
            use_cpu_offload = True
        else:
            use_cpu_offload = False
            
        device = self.gm.get_device_string(gpu['id'], preferred_backend=self.model_info.get("backend"))
        
        # Mappa la lingua dell'app al codice lingua di Kokoro
        lang_map = {
            "english": "a", "en": "a",
            "italian": "i", "it": "i",
            "spanish": "e", "es": "e",
            "french": "f", "fr": "f",
        }
        app_language = kwargs.get("language", "english").lower()
        kokoro_lang = lang_map.get(app_language, "a") # Fallback a inglese
        
        if app_language not in lang_map:
            logger.warning(f"Lingua {app_language} non supportata nativamente da Kokoro. Uso fallback su inglese ('a').")
            
        if self.model is None:
            logger.info("Caricamento modello Kokoro TTS...")
            try:
                logger.info("Kokoro: Inizializzazione KModel...")
                # Forza float32 per evitare problemi di stabilità con ROCm/FP16
                self.model = KModel(disable_complex=True).to(device).float()
                logger.info(f"Kokoro: KModel caricato su {device} con dtype float32.")
                logger.info("Kokoro: Inizializzazione KPipeline...")
                self.pipeline = KPipeline(model=self.model, lang_code=kokoro_lang)
                logger.info("Kokoro: KPipeline inizializzata.")
            except Exception as e:
                logger.exception(f"Errore nel caricamento del modello Kokoro.")
                raise
                
        import re
        # Pulizia del testo da markdown, asterischi e processi di pensiero residui
        text = re.sub(r'\<think\>.*?\<\/think\>', '', text, flags=re.DOTALL).strip()
        text = re.sub(r'^.*?(Here\'s a thinking process:|Thinking Process:).*?\n', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\*+', '', text).strip()
        text = re.sub(r'#+', '', text).strip()
        # Rimozione di altri caratteri speciali che il TTS potrebbe leggere
        text = re.sub(r'[\[\]\(\)\{\}<>\\|~`^]', '', text).strip()
        text = re.sub(r'\s+', ' ', text).strip()
        
        logger.info(f"Generazione voce per testo: {text}")
        
        # Mappa la lingua a una lista di voci Kokoro disponibili
        voice_map = {
            "english": ["af_heart", "af_bella", "af_sky", "am_adam", "am_michael"],
            "italian": ["im_nicola", "if_sara"],
            "spanish": ["ef_dora", "em_alex"],
            "french": ["ff_siwis"]
        }
        available_voices = voice_map.get(app_language, ["af_heart"])
        voice_name = kwargs.get("voice_name", random.choice(available_voices))
        
        # Variazione casuale della velocità per cambiare leggermente la tonalità
        speed = random.uniform(0.95, 1.05)
        
        import numpy as np
        
        with torch.no_grad():
            audio_chunks = []
            logger.info("Kokoro: Avvio generazione audio (pipeline)...")
            for i, (graphemes, phonemes, audio) in enumerate(self.pipeline(text, voice=voice_name, speed=speed)):
                logger.info(f"Kokoro: Chunk {i} generato.")
                audio_chunks.append(audio.cpu().numpy().squeeze())
            logger.info("Kokoro: Generazione audio completata.")
                
        if not audio_chunks:
            raise RuntimeError("Kokoro non ha generato alcun audio.")
            
        audio_data = np.concatenate(audio_chunks)
        
        # Applicazione di un equalizzatore casuale a 3 bande (basso, medio, alto)
        try:
            sr = 24000
            # Banda bassa
            low_cutoff = random.uniform(100, 500)
            b_low, a_low = butter(2, low_cutoff / (sr / 2), btype='low')
            low_part = lfilter(b_low, a_low, audio_data)
            
            # Banda alta
            high_cutoff = random.uniform(2000, 5000)
            b_high, a_high = butter(2, high_cutoff / (sr / 2), btype='high')
            high_part = lfilter(b_high, a_high, audio_data)
            
            # Banda media (resto del segnale)
            mid_part = audio_data - low_part - high_part
            
            # Guadagni casuali per ogni banda
            low_gain = random.uniform(0.5, 1.5)
            mid_gain = random.uniform(0.8, 1.2)
            high_gain = random.uniform(0.5, 1.5)
            
            audio_data = low_part * low_gain + mid_part * mid_gain + high_part * high_gain
        except Exception as eq_e:
            logger.warning(f"Errore nell'applicazione dell'equalizzatore casuale: {eq_e}")
        
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
