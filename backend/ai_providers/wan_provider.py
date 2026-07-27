import os
import yaml
import torch
import numpy as np
from diffusers import DiffusionPipeline
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

class WanProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("video", {}).get("wan_2_1_1_3b", {})
        self.gm = GPUManager()
        self.pipeline = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompt: str, output_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Wan 2.2 non installato.")

        gpu = self.gm.get_gpu_for_task("video_generation", self.get_gpu_requirements().get("vram_required_gb", 0))
        if not gpu:
            logger.warning("Nessuna GPU con VRAM sufficiente per Wan 2.2. Uso GPU con offload su RAM.")
            gpu = self.gm.get_gpu_for_task_ignore_vram("video_generation")
            if not gpu:
                raise RuntimeError("Nessuna GPU assegnata per la video generation.")
            use_cpu_offload = True
        else:
            use_cpu_offload = False

        device = self.gm.get_device_string(gpu['id'], preferred_backend=self.model_info.get("backend"))

        if self.pipeline is None:
            logger.info("Caricamento pipeline Wan 2.2...")
            model_path = self.model_info.get("path")
            try:
                self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
                self.pipeline.enable_attention_slicing()
                if use_cpu_offload:
                    self.pipeline.enable_model_cpu_offload(device=device)
                else:
                    self.pipeline.to(device)
            except Exception as e:
                logger.exception(f"Errore nel caricamento del modello su GPU. Fallback con offload su RAM.")
                if self.pipeline is not None:
                    self.pipeline.to("cpu")
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                available_ram = self.gm.get_available_system_ram_gb()
                if available_ram < 8.0:
                    raise RuntimeError(f"RAM di sistema insufficiente ({available_ram:.2f}GB) per il fallback su CPU. Operazione annullata per evitare il blocco del sistema.")
                
                logger.warning(f"RAM disponibile: {available_ram:.2f}GB. Uso sequential CPU offload per evitare OOM.")
                self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_sequential_cpu_offload(device=device)

        logger.info(f"Generazione video per prompt: {prompt}")
        # Aggiungi parametri per video verticale (Shorts)
        video = self.pipeline(
            prompt, 
            num_inference_steps=30, 
            height=960, 
            width=544,
            num_frames=49
        ).frames[0]

        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()

        # Converti da [0, 1] float32 a [0, 255] uint8
        video = (video * 255).round().astype("uint8")

        import imageio
        imageio.mimsave(output_path, video, fps=24)
        logger.info(f"Video salvato in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "video", "model": "wan_2_1_1_3b"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}

    def cleanup(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
