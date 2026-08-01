import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
os.environ["TORCH_BLAS_PREFER_HIPBLAS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512"
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

def _load_model_from_safetensors_streaming(model_class, subfolder, base_path, torch_dtype=torch.float16):
    """
    Load a model from a directory containing safetensors files.
    Streams weights directly into a float16 model and drops page cache after each file.
    """
    config = model_class.load_config(base_path, subfolder=subfolder)
    model = model_class.from_config(config, torch_dtype=torch_dtype)
    safetensors_files = sorted(glob.glob(os.path.join(base_path, subfolder, "*.safetensors")))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {os.path.join(base_path, subfolder)}")
    param_dict = dict(model.named_parameters())
    for sf in safetensors_files:
        file_obj = open(sf, "rb")
        fd = file_obj.fileno()
        try:
            with safe_open(file_obj, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    tensor = tensor.to(torch_dtype)
                    if key in param_dict:
                        param_dict[key].data.copy_(tensor)
                    else:
                        logger.warning(f"Key {key} not found in {model_class.__name__}, skipping.")
        finally:
            # Drop page cache for this file
            try:
                os.posix_fadvise(fd, 0, os.fstat(fd).st_size, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass
            file_obj.close()
    del param_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    return model

class WanProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("video", {}).get("wan_2_2_14b", {})
        self.offload_strategy = self.model_info.get("offload_strategy", "model")
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
            logger.info("Caricamento pipeline Wan 2.2 14B (Img2Video) da FP8...")
            model_path = os.path.abspath(self.model_info.get("path"))
            base_model_path = self.model_info.get("base_model_path", "Wan-AI/Wan2.2-TI2V-14B-Diffusers")
            
            # Load transformer config only (no weights), then inject FP8 state dict
            config = WanTransformer3DModel.load_config(base_model_path, subfolder="transformer")
            transformer = WanTransformer3DModel.from_config(config, torch_dtype=torch.float16)
            
            # Stream tensors from safetensors file directly into transformer parameters
            logger.info(f"Iniezione pesi FP8 nel transformer (streaming)...")
            from safetensors import safe_open
            model_file_obj = open(model_path, "rb")
            model_fd = model_file_obj.fileno()
            try:
                with safe_open(model_file_obj, framework="pt", device="cpu") as f:
                    param_dict = dict(transformer.named_parameters())
                    for key in f.keys():
                        if key == "scaled_fp8":
                            continue
                        tensor = f.get_tensor(key)
                        tensor = tensor.to(torch.float16)
                        if key in param_dict:
                            param_dict[key].data.copy_(tensor)
                        else:
                            logger.warning(f"Key {key} not found in transformer, skipping.")
                del param_dict
            finally:
                # Drop page cache for the FP8 file
                try:
                    os.posix_fadvise(model_fd, 0, os.fstat(model_fd).st_size, os.POSIX_FADV_DONTNEED)
                except Exception:
                    pass
                model_file_obj.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            from transformers import T5EncoderModel, T5Tokenizer
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

            logger.info("Caricamento VAE (streaming float16)...")
            vae = _load_model_from_safetensors_streaming(
                AutoencoderKLWan, "vae", base_model_path, torch_dtype=torch.float16
            )
            self.ram()

            logger.info("Caricamento text encoder T5 (streaming float16)...")
            text_encoder = _load_model_from_safetensors_streaming(
                T5EncoderModel, "text_encoder", base_model_path, torch_dtype=torch.float16
            )
            tokenizer = T5Tokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
            self.ram()

            logger.info("Caricamento image encoder CLIP (streaming float16)...")
            image_encoder = _load_model_from_safetensors_streaming(
                CLIPVisionModelWithProjection, "image_encoder", base_model_path, torch_dtype=torch.float16
            )
            image_processor = CLIPImageProcessor.from_pretrained(base_model_path, subfolder="image_encoder")
            self.ram()

            logger.info(f"Caricamento pipeline da {base_model_path}...")
            self.pipeline = WanImageToVideoPipeline.from_pretrained(
                base_model_path,
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                image_encoder=image_encoder,
                image_processor=image_processor,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )

            self.ram()  # log current RAM usage

            # self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            #     self.pipeline.scheduler.config,
            #     use_karras_sigmas=True
            # )
            # logger.info("Scheduler replaced with DPMSolverMultistepScheduler (Karras sigmas).")

            # Try model offload first; if OOM occurs, fall back to sequential offload
            if self.offload_strategy == "sequential":
                self.pipeline.enable_sequential_cpu_offload()
                logger.info("Enabled sequential CPU offload (from config).")
            else:
                self.pipeline.enable_model_cpu_offload()
                logger.info("Enabled model CPU offload.")

            # Enable additional memory-saving features
            slicing_enabled = False
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
                slicing_enabled = True
                logger.info("VAE slicing enabled via pipeline.")
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
                logger.info("Attention slicing enabled.")
            if hasattr(self.pipeline, "vae"):
                if hasattr(self.pipeline.vae, "enable_slicing") and not slicing_enabled:
                    self.pipeline.vae.enable_slicing()
                    logger.info("VAE slicing enabled directly on VAE.")

            # Enable memory-efficient attention for ROCm
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                logger.info("Memory-efficient SDP enabled.")
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    logger.info("Flash SDP enabled.")
                except Exception as e:
                    logger.warning(f"Flash SDP not available: {e}")

            logger.info(f"Pipeline caricata. Transformer dtype: {self.pipeline.transformer.dtype}")

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

            # Attempt generation; if OOM, switch to sequential offload and retry once
            try:
                with torch.inference_mode():
                    output = self.pipeline(
                        image=current_image,
                        prompt=prompt,
                        negative_prompt="blurry, distorted, glitchy, low quality, bad anatomy, watermark, text, deformed, mutated, extra limbs, bad framing",
                        num_inference_steps=steps,
                        num_frames=frames_per_clip,
                        height=height,
                        width=width,
                        guidance_scale=3.5,
                        generator=generator,
                        output_type="pil",
                        callback_on_step_end=progress_callback,
                        callback_on_step_end_tensor_inputs=[]
                    ).frames[0]
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM with model offload, switching to sequential CPU offload and retrying...")
                    # Clean up and re-enable with sequential offload
                    _transformer = self.pipeline.transformer
                    _vae = self.pipeline.vae
                    del self.pipeline
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Rebuild pipeline with sequential offload
                    _base_model_path = self.model_info.get("base_model_path", "Wan-AI/Wan2.2-TI2V-14B-Diffusers")
                    self.pipeline = WanImageToVideoPipeline.from_pretrained(
                        _base_model_path,
                        transformer=_transformer,
                        vae=_vae,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        use_safetensors=True
                    )
                    self.pipeline.enable_sequential_cpu_offload()
                    # Re-apply slicing if possible
                    if hasattr(self.pipeline, "enable_vae_slicing"):
                        self.pipeline.enable_vae_slicing()
                    if hasattr(self.pipeline, "enable_attention_slicing"):
                        self.pipeline.enable_attention_slicing()
                    if hasattr(self.pipeline, "vae"):
                        if hasattr(self.pipeline.vae, "enable_slicing"):
                            self.pipeline.vae.enable_slicing()

                    # Enable memory-efficient attention for ROCm
                    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                        torch.backends.cuda.enable_mem_efficient_sdp(True)
                        logger.info("Memory-efficient SDP enabled (retry).")
                    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                        try:
                            torch.backends.cuda.enable_flash_sdp(True)
                            logger.info("Flash SDP enabled (retry).")
                        except Exception as e:
                            logger.warning(f"Flash SDP not available: {e}")

                    gc.collect()
                    torch.cuda.empty_cache()
                    with torch.inference_mode():
                        output = self.pipeline(
                            image=current_image,
                            prompt=prompt,
                            negative_prompt="blurry, distorted, glitchy, low quality, bad anatomy, watermark, text, deformed, mutated, extra limbs, bad framing",
                            num_inference_steps=steps,
                            num_frames=frames_per_clip,
                            height=height,
                            width=width,
                            guidance_scale=3.5,
                            generator=generator,
                            output_type="pil",
                            callback_on_step_end=progress_callback,
                            callback_on_step_end_tensor_inputs=[]
                        ).frames[0]
                else:
                    raise

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
