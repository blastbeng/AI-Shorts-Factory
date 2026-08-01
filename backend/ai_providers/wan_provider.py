import os
import gc
import json
import shutil
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
import imageio
import subprocess as sp
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

torch.set_num_threads(1)  # reduce CPU memory overhead from parallel operations


def _cleanup_base_model_dir(base_dir):
    if not os.path.isdir(base_dir):
        return

    # 1. Remove any transformer_* folders except 'transformer'
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("transformer_") and entry != "transformer":
            shutil.rmtree(full_path, ignore_errors=True)

    # 2. Remove any weight files inside transformer/ (keep only config.json)
    transformer_dir = os.path.join(base_dir, "transformer")
    if os.path.isdir(transformer_dir):
        for entry in os.listdir(transformer_dir):
            if entry != "config.json":
                full_path = os.path.join(transformer_dir, entry)
                if os.path.isfile(full_path):
                    os.remove(full_path)

    # 3. Fix model_index.json: ensure only a single 'transformer' key exists
    model_index_path = os.path.join(base_dir, "model_index.json")
    if not os.path.exists(model_index_path):
        raise FileNotFoundError(f"model_index.json missing in {base_dir}")

    with open(model_index_path, "r") as f:
        model_index = json.load(f)

    changed = False
    for k in list(model_index.keys()):
        if k.startswith("transformer_") and k != "transformer":
            del model_index[k]
            changed = True

    if model_index.get("transformer") != "transformer":
        model_index["transformer"] = "transformer"
        changed = True

    if changed:
        with open(model_index_path, "w") as f:
            json.dump(model_index, f, indent=2)


class WanProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("video", {}).get("wan_2_2_14b", {})
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

    def _load_pipeline(self):
        if self.pipeline is not None:
            return

        preferred_backend = self.model_info.get("backend")
        gpu = self.gm.get_gpu_for_task("video_generation", self.get_gpu_requirements().get("vram_required_gb", 0), preferred_backend=preferred_backend)
        if not gpu:
            logger.warning("Nessuna GPU con VRAM sufficiente per Wan. Uso GPU con offload su RAM.")
            gpu = self.gm.get_gpu_for_task_ignore_vram("video_generation", preferred_backend=preferred_backend)
            if not gpu:
                raise RuntimeError("Nessuna GPU assegnata per la video generation.")

        device = self.gm.get_device_string(gpu['id'], preferred_backend=preferred_backend)

        _cleanup_base_model_dir(self.base_model_path)

        logger.info("Loading base Wan pipeline...")
        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            transformer=None,
        )
        if self.pipeline.transformer is not None:
            del self.pipeline.transformer
            self.pipeline.transformer = None

        transformer_config_path = os.path.join(self.base_model_path, "transformer")
        if not os.path.exists(os.path.join(transformer_config_path, "config.json")):
            raise FileNotFoundError(f"Transformer config not found at {transformer_config_path}/config.json")

        logger.info(f"Loading FP8 transformer from {self.model_path} ...")
        transformer_dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else "auto"
        transformer = WanTransformer3DModel.from_single_file(
            self.model_path,
            config=transformer_config_path,
            torch_dtype=transformer_dtype,
        )
        self.pipeline.transformer = transformer

        self.pipeline.to(device)

        if hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()
        if hasattr(self.pipeline, "enable_vae_tiling"):
            self.pipeline.enable_vae_tiling()
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()

    def generate(self, prompts: list, output_path: str, *args, frames_per_clip=int(os.getenv("GEN_FRAMES", 49)), width=int(os.getenv("GEN_WIDTH", 256)), height=int(os.getenv("GEN_HEIGHT", 448)), steps=int(os.getenv("GEN_WAN_STEPS", 40)), **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Wan 2.2 14B non installato.")

        job_id = kwargs.get("job_id")
        image_path = kwargs.get("image_path")

        self._load_pipeline()

        from backend.services.progress_tracker import ProgressTracker
        tracker = ProgressTracker()

        fps = 24.0
        temp_clips = []
        last_frame = None
        first_clip_frames = None

        for i, prompt_data in enumerate(prompts):
            img_prompt, vid_prompt = prompt_data
            if i == 0:
                prompt = f"{img_prompt}. {vid_prompt}, high quality, realistic, smooth motion, 4k"
            else:
                prompt = f"{vid_prompt}, smooth transition, high quality, realistic, smooth motion, 4k"

            def progress_callback(pipe, step, timestep, callback_kwargs):
                logger.info(f"Wan generation progress (clip {i+1}): step {step + 1}/{steps}")
                if job_id:
                    tracker.update(job_id, "video_generation", step + 1, steps, f"Generazione clip {i+1}/{len(prompts)}: step {step + 1}/{steps}")
                return callback_kwargs

            # Prepare conditioning image
            if i == 0 and image_path and os.path.exists(image_path):
                init_image = Image.open(image_path).convert("RGB")
                init_image = init_image.resize((width, height), Image.LANCZOS)
                current_image = init_image
            elif i > 0 and last_frame is not None:
                frame = cv2.resize(last_frame, (width, height), interpolation=cv2.INTER_CUBIC)
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).clip(0, 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                current_image = Image.fromarray(frame)
            else:
                current_image = Image.new("RGB", (width, height), color="black")

            seed = self.base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(seed)

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
                    callback_on_step_end_tensor_inputs=[],
                ).frames[0]

            actual_frames = len(output)
            if first_clip_frames is None:
                first_clip_frames = actual_frames

            # Save clip
            temp_clip_path = output_path.replace(".mp4", f"_clip_{i}.mp4")
            writer = imageio.get_writer(
                temp_clip_path,
                fps=fps,
                codec="libx264",
                ffmpeg_params=["-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"],
            )
            for frame in output:
                arr = np.array(frame)
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                writer.append_data(arr)
            writer.close()
            temp_clips.append(temp_clip_path)

            # Extract last frame for next clip conditioning
            last_frame = np.array(output[-1]).copy()
            if last_frame.dtype != np.uint8:
                if last_frame.max() <= 1.0:
                    last_frame = (last_frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    last_frame = last_frame.astype(np.uint8)

            del output, current_image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate clips with crossfade
        if len(temp_clips) == 1:
            shutil.copy(temp_clips[0], output_path)
        else:
            fade_duration = 0.5
            clip_duration = first_clip_frames / fps
            offset = max(0.0, clip_duration - fade_duration - 0.1)

            filter_parts = []
            for i, clip in enumerate(temp_clips):
                filter_parts.append(f"[{i}:v]settb=AVTB,fps=24,setpts=PTS-STARTPTS[v{i}];")

            prev = "v0"
            for i in range(1, len(temp_clips)):
                next_v = f"v{i}"
                if i == len(temp_clips) - 1:
                    filter_parts.append(
                        f"[{prev}][{next_v}]xfade=transition=fade:duration={fade_duration}:offset={offset}[vout]"
                    )
                else:
                    filter_parts.append(
                        f"[{prev}][{next_v}]xfade=transition=fade:duration={fade_duration}:offset={offset}[v{i}];"
                    )
                    prev = f"v{i}"

            filter_complex = "".join(filter_parts)

            cmd = ["ffmpeg", "-y"]
            for clip in temp_clips:
                cmd.extend(["-i", clip])
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
                output_path,
            ])
            sp.run(cmd, check=True)

        # Cleanup temp clips
        for clip in temp_clips:
            if os.path.exists(clip):
                os.remove(clip)

        # Frame interpolation to 48 fps for smoother motion
        interpolated_path = output_path.replace(".mp4", "_48fps.mp4")
        sp.run([
            "ffmpeg", "-y",
            "-i", output_path,
            "-filter:v", "minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            interpolated_path,
        ], check=True)
        os.replace(interpolated_path, output_path)

        logger.info(f"Video generato con successo: {output_path}")
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
