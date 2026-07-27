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
            
        gpu = self.gm.get_gpu_for_task("audio_generation", self.get_gpu_requirements().get("vram_required_gb", 0))
        if not gpu:
            logger.warning("Nessuna GPU con VRAM sufficiente per MMAudio. Uso GPU con offload su RAM.")
            gpu = self.gm.get_gpu_for_task_ignore_vram("audio_generation")
            if not gpu:
                raise RuntimeError("Nessuna GPU assegnata per l'audio generation.")
            use_cpu_offload = True
        else:
            use_cpu_offload = False
            
        device = self.gm.get_device_string(gpu['id'], preferred_backend=self.model_info.get("backend"))
        
        if self.model is None:
            logger.info("Caricamento modello MMAudio...")
            model_path = self.model_info.get("path")
            try:
                self.processor = AutoProcessor.from_pretrained(model_path)
                if use_cpu_offload:
                    self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
                else:
                    self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
            except Exception as e:
                logger.exception(f"Errore nel caricamento del modello su GPU. Fallback con offload su RAM.")
                if self.model is not None:
                    del self.model
                    self.model = None
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                
                available_ram = self.gm.get_available_system_ram_gb()
                if available_ram < 4.0:
                    raise RuntimeError(f"RAM di sistema insufficiente ({available_ram:.2f}GB) per il fallback su CPU. Operazione annullata per evitare il blocco del sistema.")
                
                logger.warning(f"RAM disponibile: {available_ram:.2f}GB. Uso offload su RAM.")
                self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
            
        logger.info(f"Generazione audio per prompt: {prompt}")
        inputs = self.processor(text=prompt, return_tensors="pt")
        # Move inputs to the model's actual device (handles device_map="auto")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            audio_values = self.model.generate(**inputs)
            
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        audio_values = audio_values.cpu().numpy().squeeze()
        # Converti da float32 [-1, 1] a int16
        audio_values = (audio_values * 32767).astype("int16")
        wavfile.write(output_path, sampling_rate, audio_values)
        logger.info(f"Audio salvato in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "audio", "model": "mmaudio"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
