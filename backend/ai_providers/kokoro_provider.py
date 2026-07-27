import os
import yaml
import torch
import scipy.io.wavfile as wavfile
from transformers import AutoTokenizer, AutoModel
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
        
        if self.model is None:
            logger.info("Caricamento modello Kokoro TTS...")
            model_path = self.model_info.get("path")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            try:
                if use_cpu_offload:
                    self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
                else:
                    self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device)
            except Exception as e:
                logger.warning(f"Errore nel caricamento del modello su GPU ({e}). Fallback con offload su RAM.")
                if self.model is not None:
                    del self.model
                    self.model = None
                    import gc
                    import torch
                    gc.collect()
                    torch.cuda.empty_cache()
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
            
        logger.info(f"Generazione voce per testo: {text}")
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            audio_output = self.model.generate(**inputs, max_new_tokens=1000)
            
        # Assumendo che il modello restituisca un tensore audio
        # La logica di salvataggio dipende dall'output specifico del modello Kokoro
        audio_data = audio_output.cpu().numpy().squeeze()
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
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
