import os
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import LTXImageToVideoPipeline
from transformers import T5EncoderModel
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger
import imageio

class LtxProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("video", {}).get("ltx_video", {})
        self.gm = GPUManager()
        self.pipeline = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompts: list, output_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello LTX Video non installato.")
            
        job_id = kwargs.get("job_id")
        image_path = kwargs.get("image_path")
        target_duration = kwargs.get("target_duration")
        
        fps = 24.0
        frames_per_clip = 49
        seconds_per_clip = frames_per_clip / fps  # ~2.04 seconds

        if target_duration and target_duration > 0:
            num_clips_needed = max(1, int(np.ceil(target_duration / seconds_per_clip)))
            # Extend or trim prompts to match needed clip count
            if len(prompts) < num_clips_needed:
                # Repeat last prompt for remaining clips
                prompts = prompts + [prompts[-1]] * (num_clips_needed - len(prompts))
            elif len(prompts) > num_clips_needed:
                prompts = prompts[:num_clips_needed]
        
        preferred_backend = self.model_info.get("backend")
        gpu = self.gm.get_gpu_for_task_ignore_vram("video_generation", preferred_backend=preferred_backend)
        if not gpu:
            raise RuntimeError("Nessuna GPU assegnata per la video generation.")
        
        if self.pipeline is None:
            logger.info("Caricamento pipeline LTX Video (Img2Video)...")
            model_path = os.path.abspath(self.model_info.get("path"))
            
            text_encoder = T5EncoderModel.from_pretrained(
                "google/t5-v1_1-xxl",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            text_encoder.eval()
            
            self.pipeline = LTXImageToVideoPipeline.from_single_file(
                model_path,
                text_encoder=text_encoder,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            del text_encoder
            import gc
            gc.collect()
            self.pipeline.vae.enable_slicing()
            self.pipeline.vae.enable_tiling()
            self.pipeline.enable_attention_slicing("max")
            self.pipeline.enable_sequential_cpu_offload()

        import gc
        
        temp_clips = []
        last_frame = None
        for i, prompt in enumerate(prompts):
            logger.info(f"Pulizia VRAM prima della generazione clip {i+1}/{len(prompts)}...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            logger.info(f"Generazione clip {i+1}/{len(prompts)} per prompt: {prompt}")
            
            # Determine the conditioning image for this clip
            target_width = 448
            target_height = 800
            if i == 0 and image_path and os.path.exists(image_path):
                # Load and resize the initial Flux image to match video dimensions
                init_image = Image.open(image_path).convert("RGB")
                init_image = init_image.resize((target_width, target_height), Image.LANCZOS)
                current_image = init_image
            elif i > 0 and last_frame is not None:
                # Use the high-quality last frame from the previous clip in memory
                frame = np.squeeze(last_frame)
                if frame.ndim == 3:
                    current_image = Image.fromarray(frame.astype(np.uint8))
                else:
                    logger.error(f"Last frame has unexpected shape: {frame.shape}")
                    current_image = Image.new("RGB", (target_width, target_height), color="black")
            else:
                logger.warning("Nessuna immagine iniziale fornita. Uso immagine nera.")
                current_image = Image.new("RGB", (target_width, target_height), color="black")

            # Add continuity prompt for subsequent clips
            if i > 0:
                prompt = f"""Continue the previous scene seamlessly. Keep the same character and environment.
Focus on this specific action: {prompt}
Maintain dynamic camera continuity and natural motion."""

            prompt = f"""{prompt}
Cinematic video, dynamic camera following subject, natural realistic motion, smooth continuous action, professional cinematography."""

            import random
            steps = 40
            generator = torch.Generator(device="cpu").manual_seed(random.randint(0, 2**32 - 1))

            def progress_callback(pipe, step, timestep, callback_kwargs):
                logger.info(f"LTX generation progress (clip {i+1}): step {step + 1}/{steps}")
                if job_id:
                    from backend.services.progress_tracker import ProgressTracker
                    ProgressTracker().update(job_id, "video_generation", step + 1, steps, f"Generazione clip {i+1}/{len(prompts)}: step {step + 1}/{steps}")
                return callback_kwargs

            # Always generate 49 frames per clip to ensure consistent motion and audio sync
            num_frames = 49

            video = self.pipeline(
                image=current_image,
                prompt=prompt,
                num_inference_steps=steps,
                num_frames=num_frames,
                height=target_height,
                width=target_width,
                guidance_scale=20.0,
                generator=generator,
                callback_on_step_end=progress_callback,
                callback_on_step_end_tensor_inputs=[]
            ).frames[0]

            if isinstance(video, torch.Tensor):
                video = video.cpu().numpy()
            elif isinstance(video, list):
                video = np.stack([
                    np.array(frame)
                    for frame in video
                ])

            logger.info(f"LTX output type={type(video)}, shape={getattr(video, 'shape', None)}, dtype={getattr(video, 'dtype', None)}")

            # Se è float [0,1], converti
            if video.dtype != np.uint8:
                if video.max() <= 1.0:
                    video = (video * 255).round()

                video = video.astype("uint8")

            # Keep the last frame in memory for the next clip to avoid lossy compression artifacts
            last_frame = video[-1]

            temp_clip_path = output_path.replace(".mp4", f"_clip_{i}.mp4")
            writer = imageio.get_writer(temp_clip_path, fps=24.0, codec='libx264', quality=8, macro_block_size=1)
            for frame in video:
                writer.append_data(frame)
            writer.close()
            temp_clips.append(temp_clip_path)
            logger.info(f"Clip {i+1} salvata in {temp_clip_path}")
            
            # Explicitly delete large objects to prevent VRAM/RAM fragmentation
            del video
            del current_image
            if 'init_image' in locals():
                del init_image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        logger.info(f"Concatenazione di {len(temp_clips)} clip in {output_path}...")
        
        # Keep a constant 24 FPS to ensure MMAudio syncs correctly.
        # The final duration mismatch is handled by FFmpeg's -shortest flag during assembly.
        fps = 24.0
        
        final_writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8, macro_block_size=1)
        for temp_clip_path in temp_clips:
            reader = imageio.get_reader(temp_clip_path)
            for frame in reader:
                final_writer.append_data(frame)
            reader.close()
        final_writer.close()
        
        for temp_clip_path in temp_clips:
            if os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)
                
        logger.info(f"Video finale salvato in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "video", "model": "ltx_video"}

    def get_gpu_requirements(self):
        return {
            "vram_required_gb": self.model_info.get("vram_required_gb", 12),
            "backend": self.model_info.get("backend")
        }

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
