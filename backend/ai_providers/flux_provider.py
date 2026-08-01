import os
import yaml
os.environ["OMP_NUM_THREADS"]="12"
os.environ["MKL_NUM_THREADS"]="12"
os.environ["OPENBLAS_NUM_THREADS"]="12"
os.environ["VECLIB_MAXIMUM_THREADS"]="12"
os.environ["NUMEXPR_NUM_THREADS"]="12"
import torch
torch.set_num_threads(12)
torch.set_num_interop_threads(2)
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
        self.flux_gguf_path = os.path.abspath(self.model_info.get("model_path"))
        self.t5_gguf_path = self.model_info.get("t5_encoder_path", "")
        if self.t5_gguf_path:
            self.t5_gguf_path = os.path.abspath(self.t5_gguf_path)
        self.gm = GPUManager()
        self.pipeline = None
        self.text_encoder_2 = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompt: str, output_path: str, *args, width=int(os.getenv("GEN_WIDTH", 256)), height=int(os.getenv("GEN_HEIGHT", 448)), steps=int(os.getenv("GEN_FLUX_STEPS", 4)), **kwargs):
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
        
        # Check system RAM before attempting CPU offload
        available_ram = self.gm.get_available_system_ram_gb()
        if available_ram < 20.0:
            raise RuntimeError(f"RAM di sistema insufficiente ({available_ram:.2f}GB) per il fallback su CPU. Flux richiede circa 20GB di RAM. Operazione annullata per evitare il blocco del sistema.")
        
        device = self.gm.get_device_string(
            gpu['id'],
            preferred_backend=preferred_backend
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        if self.pipeline is None:
            logger.info("Caricamento pipeline Flux GGUF...")
            try:
                quant_config = GGUFQuantizationConfig(
                    compute_dtype=torch.float16
                )

                transformer = FluxTransformer2DModel.from_single_file(
                    self.flux_gguf_path,
                    quantization_config=quant_config
                )
                
                pipeline_kwargs = {
                    "transformer": transformer,
                    "torch_dtype": torch.float16
                }

                if self.t5_gguf_path:
                    if self.t5_gguf_path.endswith(".gguf"):
                        self.text_encoder_2 = T5EncoderModel.from_pretrained(
                            os.path.dirname(self.t5_gguf_path),
                            gguf_file=os.path.basename(self.t5_gguf_path),
                            torch_dtype=torch.float16
                        )
                    else:
                        self.text_encoder_2 = T5EncoderModel.from_pretrained(
                            self.t5_gguf_path,
                            torch_dtype=torch.float16,
                            local_files_only=True
                        )

                    self.text_encoder_2.to("cpu")
                
                self.pipeline = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    **pipeline_kwargs
                )

                self.text_encoder = self.pipeline.text_encoder
                self.tokenizer = self.pipeline.tokenizer
                self.pipeline.text_encoder_2 = self.text_encoder_2
                self.pipeline.text_encoder_2.to("cpu")
                self.pipeline.vae.enable_tiling()
                self.pipeline.vae.enable_slicing()
                self.pipeline.vae.config.force_upcast = False

                # Flash Attention and xFormers can conflict with sequential CPU offload.
                # We rely on VAE tiling/slicing and sequential offload for memory management.
                logger.info("Skipping Flash Attention/xFormers for Flux to ensure compatibility with sequential CPU offload.")

                # Determine if we need CPU offload based on VRAM
                #use_offload = gpu['vram_gb'] < 20
                #if use_offload:
                #    logger.info(
                #        f"VRAM {gpu['vram_gb']}GB < 20GB, uso model CPU offload."
                #    )
                #else:
                #    self.pipeline.transformer.to(device)
                #    self.pipeline.vae.to(device)

                logger.info(f"DEVICE SELECTED = {device}")
                logger.info(f"CUDA AVAILABLE = {torch.cuda.is_available()}")
                logger.info(f"GPU = {gpu}")

                self.pipeline.transformer.to("cuda:0")
                self.pipeline.vae.to("cuda:0")
                
                logger.info(
                    f"Transformer param device: {next(self.pipeline.transformer.parameters()).device}"
                )

                logger.info(
                    f"Transformer: {next(self.pipeline.transformer.parameters()).device}"
                )

                logger.info(
                    f"VAE: {next(self.pipeline.vae.parameters()).device}"
                )

                logger.info(
                    f"T5: {next(self.text_encoder_2.parameters()).device}"
                )
                
            except Exception as e:
                logger.exception("Errore nel caricamento del modello Flux.")
                raise e

        print(
            "transformer dtype:",
            next(self.pipeline.transformer.parameters()).dtype
        )

        import gc
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        logger.info(f"Generazione immagine per prompt: {prompt}")
        try:
            with torch.inference_mode():

                prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt,
                    device="cpu",
                    max_sequence_length=512
                )


            flux_device = next(self.pipeline.transformer.parameters()).device

            prompt_embeds = prompt_embeds.to(
                flux_device,
                dtype=torch.float16
            )

            pooled_prompt_embeds = pooled_prompt_embeds.to(
                flux_device,
                dtype=torch.float16
            )

            image = self.pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=steps,
                guidance_scale=0.0,
                width=width,
                height=height,
            ).images[0]
        except Exception as e:
            logger.error(f"Errore durante la generazione dell'immagine: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            raise e
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

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
