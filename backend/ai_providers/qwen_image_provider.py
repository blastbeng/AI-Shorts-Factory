import os
import yaml
import torch
from diffusers import DiffusionPipeline
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

class QwenImageProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("image", {}).get("qwen_image", {})
        self.gm = GPUManager()
        self.pipeline = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompt: str, output_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Qwen Image non installato.")
            
        gpu = self.gm.get_gpu_for_task("image_generation", self.get_gpu_requirements().get("vram_required_gb", 0))
        if not gpu:
            logger.warning("Nessuna GPU con VRAM sufficiente per Qwen Image. Uso GPU con offload su RAM.")
            gpu = self.gm.get_gpu_for_task_ignore_vram("image_generation")
            if not gpu:
                raise RuntimeError("Nessuna GPU assegnata per l'image generation.")
            use_cpu_offload = True
        else:
            use_cpu_offload = False
            
        device = self.gm.get_device_string(gpu['id'], preferred_backend=self.model_info.get("backend"))
        
        if self.pipeline is None:
            logger.info("Caricamento pipeline Qwen Image...")
            model_path = self.model_info.get("path")
            try:
                self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
                if use_cpu_offload:
                    self.pipeline.enable_model_cpu_offload(device=device)
                else:
                    self.pipeline.to(device)
            except Exception as e:
                logger.warning(f"Errore nel caricamento del modello su GPU ({e}). Fallback con offload su RAM.")
                if self.pipeline is not None:
                    del self.pipeline
                    self.pipeline = None
                    import gc
                    import torch
                    gc.collect()
                    torch.cuda.empty_cache()
                self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
                self.pipeline.enable_model_cpu_offload(device=device)
            
        logger.info(f"Generazione immagine Qwen per prompt: {prompt}")
        image = self.pipeline(
            prompt, 
            num_inference_steps=30, 
            height=1920, 
            width=1080
        ).images[0]
        
        image.save(output_path)
        logger.info(f"Immagine Qwen salvata in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "image", "model": "qwen_image"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}

    def cleanup(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
