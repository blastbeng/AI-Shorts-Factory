import os
import yaml
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers import GGUFQuantizationConfig
from transformers import T5EncoderModel
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

class FluxProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("image", {}).get("flux", {})
        self.flux_gguf_path = self.model_info.get("model_path")
        self.t5_gguf_path = self.model_info.get("t5_encoder_path")
        self.gm = GPUManager()
        self.pipeline = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompt: str, output_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Flux non installato.")

        job_id = kwargs.get("job_id")
            
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
            logger.info("Caricamento pipeline Flux GGUF...")
            try:
                transformer = FluxTransformer2DModel.from_single_file(
                    self.flux_gguf_path,
                    quantization_config=GGUFQuantizationConfig(),
                    torch_dtype=torch.float16
                )
                text_encoder = T5EncoderModel.from_pretrained(
                    os.path.dirname(self.t5_gguf_path),
                    gguf_file=os.path.basename(self.t5_gguf_path),
                    torch_dtype=torch.float16
                )
                self.pipeline = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    transformer=transformer,
                    text_encoder=text_encoder,
                    torch_dtype=torch.float16
                )
                self.pipeline.enable_vae_tiling()
                self.pipeline.vae.enable_slicing()
                self.pipeline.enable_attention_slicing()
                self.pipeline.to(device)
            except Exception as e:
                logger.exception(f"Errore nel caricamento del modello su GPU. Fallback con offload su RAM.")
                if self.pipeline is not None:
                    del self.pipeline
                    self.pipeline = None
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Check system RAM before attempting CPU offload
                available_ram = self.gm.get_available_system_ram_gb()
                if available_ram < 24.0:
                    raise RuntimeError(f"RAM di sistema insufficiente ({available_ram:.2f}GB) per il fallback su CPU. Flux richiede circa 24GB di RAM. Operazione annullata per evitare il blocco del sistema.")
                
                logger.warning(f"RAM disponibile: {available_ram:.2f}GB. Uso model CPU offload per evitare OOM.")
                transformer = FluxTransformer2DModel.from_single_file(
                    self.flux_gguf_path,
                    quantization_config=GGUFQuantizationConfig(),
                    torch_dtype=torch.float16
                )
                text_encoder = T5EncoderModel.from_pretrained(
                    os.path.dirname(self.t5_gguf_path),
                    gguf_file=os.path.basename(self.t5_gguf_path),
                    torch_dtype=torch.float16
                )
                self.pipeline = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    transformer=transformer,
                    text_encoder=text_encoder,
                    torch_dtype=torch.float16
                )
                self.pipeline.enable_vae_tiling()
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_model_cpu_offload(gpu_id=gpu['device_index'])
            
        def progress_callback(pipe, step, timestep, callback_kwargs):
            logger.info(f"Flux generation progress: step {step + 1}/4")
            if job_id:
                from backend.services.progress_tracker import ProgressTracker
                ProgressTracker().update(job_id, "image_generation", step + 1, 4, f"Flux generation progress: step {step + 1}/4")
            return callback_kwargs

        logger.info(f"Generazione immagine per prompt: {prompt}")
        try:
            image = self.pipeline(
                prompt, 
                num_inference_steps=4,
                guidance_scale=0.0,
                height=640, 
                width=360,
                callback_on_step_end=progress_callback,
                callback_on_step_end_tensor_inputs=[]
            ).images[0]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM rilevato durante la generazione Flux. Pulizia completa e re-inizializzazione con sequential CPU offload...")
                
                # 1. Distruggi completamente la pipeline attuale per liberare VRAM e RAM
                if self.pipeline is not None:
                    del self.pipeline
                    self.pipeline = None
                
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                # 2. Verifica RAM disponibile
                available_ram = self.gm.get_available_system_ram_gb()
                if available_ram < 24.0:
                    raise RuntimeError(f"OOM su GPU, ma RAM di sistema insufficiente ({available_ram:.2f}GB) per il fallback su CPU. Operazione annullata per evitare il blocco del sistema.")
                
                # 3. Re-inizializza la pipeline da zero con sequential CPU offload
                logger.warning(f"RAM disponibile: {available_ram:.2f}GB. Re-inizializzazione pipeline con sequential CPU offload.")
                transformer = FluxTransformer2DModel.from_single_file(
                    self.flux_gguf_path,
                    quantization_config=GGUFQuantizationConfig(),
                    torch_dtype=torch.float16
                )
                text_encoder = T5EncoderModel.from_pretrained(
                    os.path.dirname(self.t5_gguf_path),
                    gguf_file=os.path.basename(self.t5_gguf_path),
                    torch_dtype=torch.float16
                )
                self.pipeline = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    transformer=transformer,
                    text_encoder=text_encoder,
                    torch_dtype=torch.float16
                )
                self.pipeline.enable_vae_tiling()
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_sequential_cpu_offload(gpu_id=gpu['device_index'])
                
                # 4. Riprova la generazione
                image = self.pipeline(
                    prompt, 
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    height=640, 
                    width=360,
                    callback_on_step_end=progress_callback,
                    callback_on_step_end_tensor_inputs=[]
                ).images[0]
            else:
                raise e
        
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
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force glibc to release unused memory back to the OS
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
