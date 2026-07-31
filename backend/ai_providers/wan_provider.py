import os
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
os.environ["TORCH_BLAS_PREFER_HIPBLAS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:256,roundup_power2_divisions:16"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:256,roundup_power2_divisions:16"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:256,roundup_power2_divisions:16"
os.environ["SAFETENSORS_FAST_GPU"] = "1"

import gc
import yaml
import torch
import psutil
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
import numpy as np
import cv2
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKLWan
from diffusers import WanTransformer3DModel
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger
import imageio

class WanProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("video", {}).get("wan_2_2_5b", {})
        self.gm = GPUManager()
        self.pipeline = None
        self.base_seed = 42

    def ram(self):
        p = psutil.Process(os.getpid())
        logger.info(
            f"RAM usage: {p.memory_info().rss / 1024**3:.2f} GB"
        )

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompts: list, output_path: str, *args, frames_per_clip=int(os.getenv("GEN_FRAMES", 49)), width=int(os.getenv("GEN_WIDTH", 256)), height=int(os.getenv("GEN_HEIGHT", 448)), steps=int(os.getenv("GEN_WAN_STEPS", 30)), **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Wan 2.2 5B non installato.")
            
        job_id = kwargs.get("job_id")
        image_path = kwargs.get("image_path")
        target_duration = kwargs.get("target_duration")
        
        fps = 24.0
        seconds_per_clip = frames_per_clip / fps

        preferred_backend = self.model_info.get("backend")
        gpu = self.gm.get_gpu_for_task_ignore_vram("video_generation", preferred_backend=preferred_backend)
        if not gpu:
            raise RuntimeError("Nessuna GPU assegnata per la video generation.")
        
        device = self.gm.get_device_string(gpu['id'], preferred_backend=preferred_backend)
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        
        if self.pipeline is None:
            logger.info("Caricamento pipeline Wan 2.2 5B (Img2Video) da FP8...")
            model_path = os.path.abspath(self.model_info.get("path"))
            base_model_path = self.model_info.get("base_model_path", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            
            logger.info(f"Caricamento transformer FP8 da {model_path}...")
            transformer = WanTransformer3DModel.from_single_file(
                model_path,
                config=os.path.join(base_model_path, "transformer"),
                torch_dtype=torch.float8_e4m3fn
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Caricamento VAE da {base_model_path} in float16...")
            vae = AutoencoderKLWan.from_pretrained(
                base_model_path,
                subfolder="vae",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Caricamento pipeline da {base_model_path}...")
            self.pipeline = DiffusionPipeline.from_pretrained(
                base_model_path,
                transformer=transformer,
                vae=vae,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )

            # Memory optimisations (as recommended by the model card)
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_vae_tiling()
            self.pipeline.enable_attention_slicing()
            
            # Crucial for 16GB VRAM cards like the RX 7800 XT
            self.pipeline.enable_model_cpu_offload()

            logger.info(f"Pipeline caricata. Transformer dtype: {self.pipeline.transformer.dtype}")

        temp_clips = []
        last_frame = None
        for i, prompt_data in enumerate(prompts):
            img_prompt, vid_prompt = prompt_data
            if i == 0:
                prompt = (
                    f"{img_prompt}. "
                    f"{vid_prompt}. "
                    "consistent character identity, "
                    "natural body movement, "
                    "realistic motion, "
                    "stable camera, "
                    "realistic physics"
                )
            else:
                prompt = (
                    f"{img_prompt}. "
                    f"{vid_prompt}. "
                    "Continue the exact same shot from the previous frame. "
                    "Do not change character identity. "
                    "Do not redesign the scene. "
                    "Maintain identical face, clothes, lighting and environment. "
                    "same character appearance, "
                    "same clothing, "
                    "same environment, "
                    "consistent character identity, "
                    "natural body movement, "
                    "realistic motion, "
                    "stable camera, "
                    "realistic physics"
                )

            logger.info(f"Pulizia VRAM prima della generazione clip {i+1}/{len(prompts)}...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Generazione clip {i+1}/{len(prompts)} per prompt: {prompt}")
            
            seed = self.base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(seed)
            
            # Determine the conditioning image for this clip
            target_width = width
            target_height = height
            if i == 0 and image_path and os.path.exists(image_path):
                init_image = Image.open(image_path).convert("RGB")
                init_image = init_image.resize((target_width, target_height), Image.LANCZOS)
                current_image = init_image
            elif i > 0 and last_frame is not None:
                frame = last_frame
                if frame.ndim == 3:
                    frame = cv2.resize(
                        frame,
                        (target_width, target_height),
                        interpolation=cv2.INTER_CUBIC
                    )
                    current_image = Image.fromarray(frame)
                else:
                    logger.error(f"Last frame has unexpected shape: {frame.shape}")
                    current_image = Image.new("RGB", (target_width, target_height), color="black")
            else:
                logger.warning("Nessuna immagine iniziale fornita. Uso immagine nera.")
                current_image = Image.new("RGB", (target_width, target_height), color="black")

            def progress_callback(pipe, step, timestep, callback_kwargs):
                logger.info(f"Wan generation progress (clip {i+1}): step {step + 1}/{steps}")
                if job_id:
                    from backend.services.progress_tracker import ProgressTracker
                    ProgressTracker().update(job_id, "video_generation", step + 1, steps, f"Generazione clip {i+1}/{len(prompts)}: step {step + 1}/{steps}")
                return callback_kwargs

            gc.collect()
            torch.cuda.empty_cache()

            with torch.inference_mode():
                output = self.pipeline(
                    image=current_image,
                    prompt=prompt,
                    num_inference_steps=steps,
                    num_frames=frames_per_clip,
                    height=height,
                    width=width,
                    guidance_scale=4.0,
                    generator=generator,
                    callback_on_step_end=progress_callback,
                    callback_on_step_end_tensor_inputs=[]
                ).frames[0]   # list of PIL Images

            # Extract last frame for next clip conditioning
            last_frame = np.array(output[-1]).copy()

            temp_clip_path = output_path.replace(".mp4", f"_clip_{i}.mp4")
            writer = imageio.get_writer(
                temp_clip_path,
                fps=24.0,
                codec="libx264",
                ffmpeg_params=[
                    "-crf", "16",
                    "-preset", "slow",
                    "-pix_fmt", "yuv420p"
                ]
            )
            for frame in output:
                writer.append_data(np.array(frame))
            writer.close()
            temp_clips.append(temp_clip_path)
            logger.info(f"Clip {i+1} salvata in {temp_clip_path}")
            
            del output
            del current_image
            if 'init_image' in locals():
                del init_image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"Concatenazione di {len(temp_clips)} clip in {output_path}...")
        
        list_file_path = output_path.replace(".mp4", "_list.txt")
        with open(list_file_path, "w") as f:
            for temp_clip_path in temp_clips:
                f.write(f"file '{os.path.abspath(temp_clip_path)}'\n")
        
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file_path,
            "-c", "copy",
            output_path
        ]
        subprocess.run(cmd, check=True)
        
        os.remove(list_file_path)
        for temp_clip_path in temp_clips:
            if os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)
                
        logger.info(f"Video finale salvato in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "video", "model": "wan_2_2_5b"}

    def get_gpu_requirements(self):
        return {
            "vram_required_gb": self.model_info.get("vram_required_gb", 8),
            "backend": self.model_info.get("backend")
        }

    def cleanup(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
