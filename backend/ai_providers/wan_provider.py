import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
os.environ["TORCH_BLAS_PREFER_HIPBLAS"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512"
os.environ["SAFETENSORS_FAST_GPU"] = "1"

import gc
import glob
import yaml
import torch
import psutil
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
import numpy as np
import cv2
from PIL import Image
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers import WanTransformer3DModel
from diffusers import DPMSolverMultistepScheduler
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger
from safetensors.torch import load_file as safetensors_load
from safetensors import safe_open
import imageio

torch.set_num_threads(1)  # reduce CPU memory overhead from parallel operations


class WanProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("video", {}).get("wan_2_2_14b", {})
        self.offload_strategy = self.model_info.get("offload_strategy", "sequential")
        self.gm = GPUManager()
        self.pipeline = None
        self.base_seed = 42
        # Paths
        self.model_path = os.path.abspath(self.model_info.get("path"))
        self.base_model_path = os.path.abspath(self.model_info.get("base_model_path"))

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompts: list, output_path: str, *args, frames_per_clip=int(os.getenv("GEN_FRAMES", 49)), width=int(os.getenv("GEN_WIDTH", 256)), height=int(os.getenv("GEN_HEIGHT", 448)), steps=int(os.getenv("GEN_WAN_STEPS", 40)), **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Wan 2.2 14B non installato.")
            
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
            # ---------- memory diagnostics ----------
            process = psutil.Process(os.getpid())
            ram_gb = process.memory_info().rss / 1024**3
            vram_used = torch.cuda.memory_allocated(gpu_id) / 1024**3 if torch.cuda.is_available() else 0
            logger.info(f"[MEM] Before pipeline load: RAM {ram_gb:.2f} GB, VRAM {vram_used:.2f} GB")

            logger.info("Loading base Wan pipeline (VAE, T5, CLIP, tokenizer, scheduler)...")
            # Load the full pipeline from the local base model directory.
            # low_cpu_mem_usage=True avoids keeping two copies of the weights.
            self.pipeline = WanImageToVideoPipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                transformer=None,          # <-- skip loading the default FP16 transformer
            )
            if self.pipeline.transformer is not None:
                logger.warning("Base pipeline loaded a transformer despite transformer=None. Discarding it.")
                del self.pipeline.transformer
                self.pipeline.transformer = None
            ram_gb = process.memory_info().rss / 1024**3
            vram_used = torch.cuda.memory_allocated(gpu_id) / 1024**3 if torch.cuda.is_available() else 0
            logger.info(f"[MEM] After base pipeline: RAM {ram_gb:.2f} GB, VRAM {vram_used:.2f} GB")

            # Determine the best dtype for the FP8 transformer
            if hasattr(torch, "float8_e4m3fn"):
                transformer_dtype = torch.float8_e4m3fn
                logger.info("Using torch.float8_e4m3fn for transformer.")
            else:
                transformer_dtype = "auto"  # safetensors will keep original FP8
                logger.warning("torch.float8_e4m3fn not available, loading transformer with dtype='auto' (FP8 preserved).")

            logger.info(f"Loading FP8 transformer from {self.model_path} ...")
            transformer = WanTransformer3DModel.from_single_file(
                self.model_path,
                config=self.pipeline.transformer.config,
                torch_dtype=transformer_dtype,
            )
            # Replace the pipeline's transformer with the FP8 one.
            self.pipeline.transformer = transformer

            ram_gb = process.memory_info().rss / 1024**3
            vram_used = torch.cuda.memory_allocated(gpu_id) / 1024**3 if torch.cuda.is_available() else 0
            logger.info(f"[MEM] After FP8 transformer: RAM {ram_gb:.2f} GB, VRAM {vram_used:.2f} GB")

            # Enable sequential CPU offload – best for low VRAM.
            self.pipeline.enable_sequential_cpu_offload()
            logger.info("Sequential CPU offload enabled.")

            # Memory‑saving features
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
                logger.info("VAE slicing enabled.")
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
                logger.info("Attention slicing enabled.")

            # ROCm memory‑efficient attention (if available)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                logger.info("Memory-efficient SDP enabled.")
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    logger.info("Flash SDP enabled.")
                except Exception as e:
                    logger.warning(f"Flash SDP not available: {e}")

            ram_gb = process.memory_info().rss / 1024**3
            vram_used = torch.cuda.memory_allocated(gpu_id) / 1024**3 if torch.cuda.is_available() else 0
            logger.info(f"[MEM] Pipeline ready: RAM {ram_gb:.2f} GB, VRAM {vram_used:.2f} GB")

        temp_clips = []
        last_frame = None
        for i, prompt_data in enumerate(prompts):
            img_prompt, vid_prompt = prompt_data
            # Build a consistent scene description
            if i == 0:
                prompt = f"{img_prompt}. {vid_prompt}, high quality, realistic, smooth motion, 4k"
            else:
                prompt = f"{vid_prompt}, smooth transition, high quality, realistic, smooth motion, 4k"

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
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).clip(0, 255)
                        frame = frame.astype(np.uint8)
                    current_image = Image.fromarray(frame)
                else:
                    logger.error(f"Last frame has unexpected shape: {frame.shape}")
                    current_image = Image.new("RGB", (target_width, target_height), color="black")
            else:
                logger.warning("Nessuna immagine iniziale fornita. Uso immagine nera.")
                current_image = Image.new("RGB", (target_width, target_height), color="black")

            logger.info(f"Conditioning image for clip {i+1}: size={current_image.size}, mode={current_image.mode}")
            # Quick sanity check: log min/max pixel values
            img_arr = np.array(current_image)
            logger.info(f"Image pixel range: min={img_arr.min()}, max={img_arr.max()}, mean={img_arr.mean():.1f}")

            def progress_callback(pipe, step, timestep, callback_kwargs):
                logger.info(f"Wan generation progress (clip {i+1}): step {step + 1}/{steps}")
                if job_id:
                    from backend.services.progress_tracker import ProgressTracker
                    ProgressTracker().update(job_id, "video_generation", step + 1, steps, f"Generazione clip {i+1}/{len(prompts)}: step {step + 1}/{steps}")
                return callback_kwargs

            gc.collect()
            torch.cuda.empty_cache()

            # Memory before generation
            process = psutil.Process(os.getpid())
            ram_before = process.memory_info().rss / 1024**3
            vram_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            logger.info(f"[MEM] Before clip {i+1} generation: RAM {ram_before:.2f} GB, VRAM {vram_before:.2f} GB")

            # Attempt generation; if OOM, halve frames and retry once
            current_frames = frames_per_clip
            for attempt in range(2):
                try:
                    with torch.inference_mode():
                        output = self.pipeline(
                            image=current_image,
                            prompt=prompt,
                            negative_prompt="blurry, distorted, glitchy, low quality, bad anatomy, watermark, text, deformed, mutated, extra limbs, bad framing",
                            num_inference_steps=steps,
                            num_frames=current_frames,
                            height=height,
                            width=width,
                            guidance_scale=3.5,
                            generator=generator,
                            output_type="pil",
                            callback_on_step_end=progress_callback,
                            callback_on_step_end_tensor_inputs=[]
                        ).frames[0]
                    break  # success, exit retry loop
                except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower() and attempt == 0:
                        logger.warning(f"OOM with {current_frames} frames, retrying with {current_frames // 2} frames...")
                        current_frames = max(8, current_frames // 2)
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        raise

            # Memory after generation
            ram_after = process.memory_info().rss / 1024**3
            vram_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            logger.info(f"[MEM] After clip {i+1} generation: RAM {ram_after:.2f} GB, VRAM {vram_after:.2f} GB")

            first_frame_arr = np.array(output[0])
            logger.info(f"First frame of clip {i+1}: shape={first_frame_arr.shape}, dtype={first_frame_arr.dtype}, min={first_frame_arr.min()}, max={first_frame_arr.max()}")

            # Extract last frame for next clip conditioning
            last_frame = np.array(output[-1]).copy()
            if last_frame.dtype != np.uint8:
                if last_frame.max() <= 1.0:
                    last_frame = (last_frame * 255).clip(0, 255)
                last_frame = last_frame.astype(np.uint8)

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
                arr = np.array(frame)
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                writer.append_data(arr)
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

        logger.info(f"Concatenazione di {len(temp_clips)} clip con crossfade in {output_path}...")
        
        import subprocess
        import shutil

        if len(temp_clips) == 1:
            # Single clip, just copy
            shutil.copy(temp_clips[0], output_path)
        else:
            # Build ffmpeg filter chain for crossfade
            fade_duration = 0.5
            clip_duration = frames_per_clip / fps
            # offset must leave enough time for the fade to finish before the first clip ends
            offset = max(0.0, clip_duration - fade_duration - 0.1)

            filter_parts = []
            for i, clip in enumerate(temp_clips):
                filter_parts.append(f"[{i}:v]settb=AVTB,fps=24,setpts=PTS-STARTPTS[v{i}];")

            prev = "v0"
            for i in range(1, len(temp_clips)):
                next_v = f"v{i}"
                if i == len(temp_clips) - 1:
                    # Last crossfade: output to [vout]
                    filter_parts.append(f"[{prev}][{next_v}]xfade=transition=fade:duration={fade_duration}:offset={offset}[vout]")
                else:
                    filter_parts.append(f"[{prev}][{next_v}]xfade=transition=fade:duration={fade_duration}:offset={offset}[v{i}];")
                    prev = f"v{i}"
            
            filter_complex = "".join(filter_parts)
            
            cmd = [
                "ffmpeg", "-y",
            ]
            for clip in temp_clips:
                cmd.extend(["-i", clip])
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
                output_path
            ])
            subprocess.run(cmd, check=True)

        for temp_clip_path in temp_clips:
            if os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)

        # Frame interpolation to 48 fps for smoother motion
        interpolated_path = output_path.replace(".mp4", "_48fps.mp4")
        cmd_interp = [
            "ffmpeg", "-y",
            "-i", output_path,
            "-filter:v", "minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            interpolated_path
        ]
        subprocess.run(cmd_interp, check=True)
        # Replace original with interpolated
        os.replace(interpolated_path, output_path)
                
        logger.info(f"Video finale salvato in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "video", "model": "wan_2_2_14b"}

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
