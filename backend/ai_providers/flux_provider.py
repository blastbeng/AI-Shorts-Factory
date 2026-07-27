import os
import yaml
import torch
from diffusers import FluxPipeline
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

class FluxProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
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
            
        preferred_backend = self.model_info.get("backend")
        gpu = self.gm.get_gpu_for_task("image_generation", self.get_gpu_requirements().get("vram_required_gb", 0), preferred_backend=preferred_backend)
        if not gpu:
            logger.warning("Nessuna GPU con VRAM sufficiente per Flux. Uso GPU con offload su RAM.")
            gpu = self.gm.get_gpu_for_task_ignore_vram("image_generation", preferred_backend=preferred_backend)
            if not gpu:
                raise RuntimeError("Nessuna GPU assegnata per l'image generation.")
            use_cpu_offload = True
        else:
            use_cpu_offload = False
            
        device = self.gm.get_device_string(gpu['id'], preferred_backend=self.model_info.get("backend"))
        
        if self.pipeline is None:
            logger.info("Caricamento pipeline Flux...")
            model_path = self.model_info.get("path")
            try:
                self.pipeline = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
                self.pipeline.enable_vae_tiling()
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_attention_slicing()
                if use_cpu_offload:
                    self.pipeline.enable_model_cpu_offload(gpu_id=gpu['device_index'])
                else:
                    self.pipeline.to(device)
            except Exception as e:
                logger.exception(f"Errore nel caricamento del modello su GPU. Fallback con offload su RAM.")
                if self.pipeline is not None:
                    self.pipeline.to("cpu")
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Check system RAM before attempting CPU offload
                available_ram = self.gm.get_available_system_ram_gb()
                if available_ram < 8.0:
                    raise RuntimeError(f"RAM di sistema insufficiente ({available_ram:.2f}GB) per il fallback su CPU. Operazione annullata per evitare il blocco del sistema.")
                
                logger.warning(f"RAM disponibile: {available_ram:.2f}GB. Uso sequential CPU offload per evitare OOM.")
                self.pipeline = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
                self.pipeline.enable_vae_tiling()
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_sequential_cpu_offload(gpu_id=gpu['device_index'])
            
        logger.info(f"Generazione immagine per prompt: {prompt}")
        image = self.pipeline(
            prompt, 
            num_inference_steps=4,
            guidance_scale=0.0,
            height=960, 
            width=544
        ).images[0]
        
        image.save(output_path)
        logger.info(f"Immagine salvata in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "image", "model": "flux"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}

    def cleanup(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
