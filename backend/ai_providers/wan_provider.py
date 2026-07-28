import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

        torch.backends.cudnn.benchmark = True

        # Log PyTorch's view of the GPUs
        if torch.cuda.is_available():
            logger.info(f"PyTorch vede {torch.cuda.device_count()} dispositivi CUDA:")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
        else:
            logger.warning("PyTorch non rileva dispositivi CUDA. Verifica l'installazione di PyTorch con supporto ROCm.")

        preferred_backend = self.model_info.get("backend", "rocm")
        gpu = self.gm.get_gpu_for_task("video_generation", self.get_gpu_requirements().get("vram_required_gb", 0), preferred_backend=preferred_backend)
        if not gpu:
            logger.warning("Nessuna GPU con VRAM sufficiente per Wan 2.2. Uso GPU con offload su RAM.")
            gpu = self.gm.get_gpu_for_task_ignore_vram("video_generation", preferred_backend=preferred_backend)
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
                try:
                    logger.info("Caricamento pipeline Wan 2.2 con PyTorch SDPA (Flash Attention nativo)...")
                    self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
                except Exception as attn_e:
                    logger.warning(f"SDPA non disponibile, caricamento standard: {attn_e}")
                    self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
                # Rimuoviamo enable_attention_slicing() per velocizzare, se va in OOM il fallback lo riattiverà
                if hasattr(self.pipeline, "enable_vae_tiling"):
                    self.pipeline.enable_vae_tiling()
                if hasattr(self.pipeline, "enable_vae_slicing"):
                    self.pipeline.enable_vae_slicing()
                self.pipeline.enable_attention_slicing()
                
                logger.info("Torch compile disabilitato per VRAM limitata")
                logger.info("Uso model CPU offload per evitare OOM (più veloce del sequential).")
                self.pipeline.enable_model_cpu_offload(gpu_id=gpu['device_index'])
            except Exception as e:
                logger.exception(f"Errore nel caricamento del modello su GPU. Fallback con offload su RAM.")
                if self.pipeline is not None:
                    self.pipeline.to("cpu")
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                available_ram = self.gm.get_available_system_ram_gb()
                if available_ram < 8.0:
                    raise RuntimeError(f"RAM di sistema insufficiente ({available_ram:.2f}GB) per il fallback su CPU. Operazione annullata per evitare il blocco del sistema.")
                
                logger.warning(f"RAM disponibile: {available_ram:.2f}GB. Uso model CPU offload per evitare OOM (più veloce del sequential).")
                try:
                    self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
                except Exception:
                    self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
                self.pipeline.enable_attention_slicing()
                if hasattr(self.pipeline, "enable_vae_slicing"):
                    self.pipeline.enable_vae_slicing()
                self.pipeline.enable_model_cpu_offload(gpu_id=gpu['device_index'])

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        logger.info(f"Generazione video per prompt: {prompt}")
        
        def progress_callback(pipe, step, timestep, callback_kwargs):
            logger.info(f"Wan generation progress: step {step + 1}/20")
            return callback_kwargs

        # Aggiungi parametri per video verticale (Shorts)
        video = self.pipeline(
            prompt, 
            num_inference_steps=20, 
            height=832, 
            width=480,
            num_frames=16,
            callback_on_step_end=progress_callback,
            callback_on_step_end_tensor_inputs=[]
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
